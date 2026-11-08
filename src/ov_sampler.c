/* Sampler de difusão mascarada (ver ov_sampler.h). */

#include "ov_sampler.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

ov_sample_config ov_sample_config_default(void) {
    ov_sample_config c;
    c.num_step = 32;
    c.guidance_scale = 2.0f;
    c.t_shift = 0.1f;
    c.layer_penalty_factor = 5.0f;
    c.position_temperature = 5.0f;
    c.class_temperature = 0.0f;
    c.seed = 42;
    return c;
}

/* xoshiro256** — RNG rápido e determinístico */
typedef struct {
    uint64_t s[4];
} rng_state;

static uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static void rng_seed(rng_state *r, uint64_t seed) {
    /* splitmix64 para espalhar a seed */
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ull;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        r->s[i] = z ^ (z >> 31);
    }
}

static uint64_t rng_next(rng_state *r) {
    uint64_t result = rotl64(r->s[1] * 5, 7) * 9;
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = rotl64(r->s[3], 45);
    return result;
}

static float rng_uniform(rng_state *r) {
    return (float)((rng_next(r) >> 11) * (1.0 / 9007199254740992.0));
}

static float gumbel_noise(rng_state *r) {
    float u = rng_uniform(r);
    return -logf(-logf(u + 1e-10f) + 1e-10f);
}

/* k-ésimo maior valor de row[0..V), via min-heap de tamanho k em `heap`
 * (k floats de espaço). É o mesmo corte de torch.topk(k) do sampler original:
 * tudo abaixo desse limiar fica fora do sorteio por classe. */
static float kth_largest(const float *row, int32_t V, int32_t k, float *heap) {
    int32_t n = 0;
    for (int32_t v = 0; v < V; v++) {
        const float x = row[v];
        if (n < k) {
            int32_t c = n++;
            heap[c] = x;
            while (c > 0) { /* sobe */
                int32_t par = (c - 1) / 2;
                if (heap[par] <= heap[c]) break;
                float tmp = heap[par];
                heap[par] = heap[c];
                heap[c] = tmp;
                c = par;
            }
        } else if (x > heap[0]) {
            heap[0] = x;
            int32_t c = 0;
            for (;;) { /* desce */
                int32_t l = 2 * c + 1, r = l + 1, sm = c;
                if (l < k && heap[l] < heap[sm]) sm = l;
                if (r < k && heap[r] < heap[sm]) sm = r;
                if (sm == c) break;
                float tmp = heap[sm];
                heap[sm] = heap[c];
                heap[c] = tmp;
                c = sm;
            }
        }
    }
    return n > 0 ? heap[0] : -INFINITY;
}

/* log_softmax in-place sobre uma linha de V floats */
static void log_softmax_row(float *row, int32_t V) {
    float mx = row[0];
    for (int32_t v = 1; v < V; v++) {
        if (row[v] > mx) mx = row[v];
    }
    double sum = 0.0;
    for (int32_t v = 0; v < V; v++) sum += exp((double)(row[v] - mx));
    float lse = mx + (float)log(sum);
    for (int32_t v = 0; v < V; v++) row[v] -= lse;
}

int ov_sample(ov_model *m, const int32_t *prefix_ids,
              const uint8_t *prefix_mask, int32_t prefix_len,
              int32_t target_len, const ov_sample_config *cfg,
              int32_t *out_tokens) {
    const ov_model_params *p = ov_model_get_params(m);
    const int32_t C = p->n_codebook;
    const int32_t V = p->audio_vocab;
    const int32_t MASK = p->audio_mask_id;
    const int32_t T = target_len;
    const int32_t cond_len = prefix_len + T;

    rng_state rng;
    rng_seed(&rng, cfg->seed);

    /* --- monta as sequências cond e uncond --- */
    int32_t *cond_ids = malloc((size_t)C * cond_len * sizeof(int32_t));
    uint8_t *cond_mask = malloc((size_t)cond_len);
    int32_t *unc_ids = malloc((size_t)C * T * sizeof(int32_t));
    uint8_t *unc_mask = malloc((size_t)T);

    for (int32_t cb = 0; cb < C; cb++) {
        memcpy(cond_ids + (size_t)cb * cond_len,
               prefix_ids + (size_t)cb * prefix_len,
               (size_t)prefix_len * sizeof(int32_t));
        for (int32_t t = 0; t < T; t++) {
            cond_ids[(size_t)cb * cond_len + prefix_len + t] = MASK;
            unc_ids[(size_t)cb * T + t] = MASK;
        }
    }
    memcpy(cond_mask, prefix_mask, (size_t)prefix_len);
    memset(cond_mask + prefix_len, 1, (size_t)T);
    memset(unc_mask, 1, (size_t)T);

    /* tokens gerados (C x T), MASK = ainda não decidido */
    int32_t *tokens = malloc((size_t)C * T * sizeof(int32_t));
    for (int32_t i = 0; i < C * T; i++) tokens[i] = MASK;

    /* --- cronograma de desmascaramento --- */
    int32_t *schedule = malloc((size_t)cfg->num_step * sizeof(int32_t));
    {
        const int32_t total = C * T;
        int32_t rem = total;
        for (int32_t s = 0; s < cfg->num_step; s++) {
            float t0 = (float)s / (float)cfg->num_step;
            float t1 = (float)(s + 1) / (float)cfg->num_step;
            t0 = cfg->t_shift * t0 / (1.0f + (cfg->t_shift - 1.0f) * t0);
            t1 = cfg->t_shift * t1 / (1.0f + (cfg->t_shift - 1.0f) * t1);
            int32_t num;
            if (s == cfg->num_step - 1) {
                num = rem;
            } else {
                num = (int32_t)ceilf((float)total * (t1 - t0));
                if (num > rem) num = rem;
            }
            schedule[s] = num;
            rem -= num;
        }
    }

    const size_t n_logit = (size_t)C * T * V;
    float *c_logits = malloc(n_logit * sizeof(float));
    float *u_logits = malloc(n_logit * sizeof(float));
    int32_t *pred = malloc((size_t)C * T * sizeof(int32_t));
    float *scores = malloc((size_t)C * T * sizeof(float));
    /* gumbel de posição sorteado em série, antes do laço paralelo */
    float *gpos = malloc((size_t)C * T * sizeof(float));

    /* sorteio por classe (class_temperature > 0): limiar de top-k por posição
     * precisa de um heap próprio por posição para o laço poder ser paralelo */
    const int32_t topk =
        cfg->class_temperature > 0.0f ? (int32_t)ceilf(0.1f * (float)V) : 0;
    float *heapbuf =
        topk > 0 ? malloc((size_t)C * T * topk * sizeof(float)) : NULL;

    int rc = -1;

    for (int32_t step = 0; step < cfg->num_step; step++) {
        int32_t k = schedule[step];
        if (k <= 0) continue;

        if (ov_model_forward(m, cond_ids, cond_mask, cond_len, T, c_logits) !=
            0) {
            goto done;
        }
        if (cfg->guidance_scale != 0.0f) {
            if (ov_model_forward(m, unc_ids, unc_mask, T, T, u_logits) != 0) {
                goto done;
            }
        }

        /* Sorteio do gumbel de posição EM SÉRIE, uma vez por posição e na
         * mesma ordem do laço original (inclusive nas posições já
         * desmascaradas, que só consomem o sorteio). Isso desacopla o RNG do
         * laço de trabalho abaixo, que então pode rodar em paralelo mantendo a
         * saída bit-idêntica à versão serial. */
        if (cfg->position_temperature > 0.0f) {
            for (int32_t i = 0; i < C * T; i++) gpos[i] = gumbel_noise(&rng);
        }

        /* CFG + previsão + confiança, por (codebook, posição). Cada posição só
         * toca a sua própria linha de logits, então o laço é paralelizável sem
         * sincronização. Posições já desmascaradas recebem -INF e nunca são
         * reescolhidas — pular o log_softmax/CFG delas economiza trabalho
         * crescente a cada passo. */
        int32_t pos; /* declarado fora: OpenMP 2.0 (MSVC) exige esta forma */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (pos = 0; pos < C * T; pos++) {
            const int32_t cb = pos / T;
            if (tokens[pos] != MASK) {
                scores[pos] = -INFINITY;
                continue;
            }
            float *crow = c_logits + (size_t)pos * V;
            log_softmax_row(crow, V);
            if (cfg->guidance_scale != 0.0f) {
                float *urow = u_logits + (size_t)pos * V;
                log_softmax_row(urow, V);
                for (int32_t v = 0; v < V; v++) {
                    crow[v] += cfg->guidance_scale * (crow[v] - urow[v]);
                }
                log_softmax_row(crow, V);
            }
            crow[MASK] = -INFINITY;

            /* confiança = máximo da linha, independente de como o token é
             * escolhido (igual ao original: log_probs.max()) */
            int32_t best = 0;
            float best_lp = crow[0];
            for (int32_t v = 1; v < V; v++) {
                if (crow[v] > best_lp) {
                    best_lp = crow[v];
                    best = v;
                }
            }

            if (cfg->class_temperature > 0.0f) {
                /* top-10% + gumbel (igual a _filter_top_k + _gumbel_sample).
                 * RNG derivado de (seed, passo, posição): independe da ordem
                 * das threads, então o resultado é reprodutível. */
                const float thr =
                    kth_largest(crow, V, topk, heapbuf + (size_t)pos * topk);
                rng_state lr;
                rng_seed(&lr, cfg->seed +
                                  0x9E3779B97F4A7C15ull *
                                      ((uint64_t)step * (uint64_t)(C * T) +
                                       (uint64_t)pos + 1ull));
                float best_g = -INFINITY;
                int32_t sampled = best;
                for (int32_t v = 0; v < V; v++) {
                    if (crow[v] < thr) continue;
                    float g = crow[v] / cfg->class_temperature +
                              gumbel_noise(&lr);
                    if (g > best_g) {
                        best_g = g;
                        sampled = v;
                    }
                }
                pred[pos] = sampled;
            } else {
                pred[pos] = best;
            }

            float sc = best_lp - (float)cb * cfg->layer_penalty_factor;
            if (cfg->position_temperature > 0.0f) {
                sc = sc / cfg->position_temperature + gpos[pos];
            }
            scores[pos] = sc;
        }

        /* top-k por score: seleção parcial simples */
        for (int32_t sel = 0; sel < k; sel++) {
            int32_t best_i = -1;
            float best_s = -INFINITY;
            for (int32_t i = 0; i < C * T; i++) {
                if (scores[i] > best_s) {
                    best_s = scores[i];
                    best_i = i;
                }
            }
            if (best_i < 0) break;
            scores[best_i] = -INFINITY;
            tokens[best_i] = pred[best_i];

            /* atualiza as entradas cond e uncond */
            int32_t cb = best_i / T;
            int32_t t = best_i % T;
            cond_ids[(size_t)cb * cond_len + prefix_len + t] = tokens[best_i];
            unc_ids[(size_t)cb * T + t] = tokens[best_i];
        }
    }

    memcpy(out_tokens, tokens, (size_t)C * T * sizeof(int32_t));
    rc = 0;

done:
    free(cond_ids);
    free(cond_mask);
    free(unc_ids);
    free(unc_mask);
    free(tokens);
    free(schedule);
    free(c_logits);
    free(u_logits);
    free(pred);
    free(scores);
    free(gpos);
    free(heapbuf);
    return rc;
}
