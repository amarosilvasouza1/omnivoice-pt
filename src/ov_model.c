/* Implementação do modelo OmniVoice sobre GGML (ver ov_model.h). */

#include "ov_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "ov_platform.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

typedef struct {
    struct ggml_tensor *attn_norm;
    struct ggml_tensor *wq, *wk, *wv, *wo;
    struct ggml_tensor *q_norm, *k_norm;
    struct ggml_tensor *ffn_norm;
    struct ggml_tensor *w_gate, *w_up, *w_down;
} ov_layer;

/* Grafo persistente para uma dada forma (seq, n_tail). Mantê-lo vivo entre
 * os passos da difusão mantém os endereços dos tensores estáveis, o que
 * permite ao backend CUDA reaproveitar o CUDA Graph capturado (replay sem
 * relançar milhares de kernels a cada passo). */
typedef struct {
    int32_t seq, n_tail;
    struct ggml_context *ctx;
    struct ggml_cgraph *gf;
    ggml_gallocr_t galloc;
    struct ggml_tensor *text_ids, *audio_ids, *amask, *pos, *logits;
    int valid;
} ov_fwd_plan;

#define OV_MAX_PLANS 2 /* cond + uncond */

struct ov_model {
    ov_model_params p;

    ggml_backend_t backend;
    ggml_backend_buffer_t weights_buf;
    struct ggml_context *weights_ctx; /* metadados dos tensores de pesos */

    struct ggml_tensor *token_embd;
    struct ggml_tensor *audio_embd;
    struct ggml_tensor *audio_heads;
    struct ggml_tensor *output_norm;
    ov_layer *layers;

    int32_t *remap; /* cópia em host de token_remap: vocab_full ids */

    ov_fwd_plan plans[OV_MAX_PLANS];
    int n_plans;
};

/* ------------------------------------------------------------------ */

static struct ggml_tensor *get_weight(struct ggml_context *ctx,
                                      const char *name) {
    struct ggml_tensor *t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "ov_model: tensor ausente no GGUF: %s\n", name);
    }
    return t;
}

static uint32_t meta_u32(const struct gguf_context *g, const char *key) {
    int64_t i = gguf_find_key(g, key);
    if (i < 0) {
        fprintf(stderr, "ov_model: chave ausente: %s\n", key);
        return 0;
    }
    return gguf_get_val_u32(g, i);
}

static float meta_f32(const struct gguf_context *g, const char *key) {
    int64_t i = gguf_find_key(g, key);
    if (i < 0) {
        fprintf(stderr, "ov_model: chave ausente: %s\n", key);
        return 0.0f;
    }
    return gguf_get_val_f32(g, i);
}

ov_model *ov_model_load(const char *gguf_path, int use_gpu) {
    struct ggml_context *meta_ctx = NULL;
    struct gguf_init_params gp = {
        /*no_alloc =*/true,
        /*ctx      =*/&meta_ctx,
    };
    struct gguf_context *g = gguf_init_from_file(gguf_path, gp);
    if (!g) {
        fprintf(stderr, "ov_model: falha ao abrir %s\n", gguf_path);
        return NULL;
    }

    ov_model *m = calloc(1, sizeof(*m));
    m->weights_ctx = meta_ctx;

    m->p.n_layer = (int32_t)meta_u32(g, "omnivoice.n_layer");
    m->p.n_embd = (int32_t)meta_u32(g, "omnivoice.n_embd");
    m->p.n_head = (int32_t)meta_u32(g, "omnivoice.n_head");
    m->p.n_head_kv = (int32_t)meta_u32(g, "omnivoice.n_head_kv");
    m->p.head_dim = (int32_t)meta_u32(g, "omnivoice.head_dim");
    m->p.n_ff = (int32_t)meta_u32(g, "omnivoice.n_ff");
    m->p.rms_norm_eps = meta_f32(g, "omnivoice.rms_norm_eps");
    m->p.rope_theta = meta_f32(g, "omnivoice.rope_theta");
    m->p.vocab_full = (int32_t)meta_u32(g, "omnivoice.vocab_size_full");
    m->p.vocab_kept = (int32_t)meta_u32(g, "omnivoice.vocab_size_kept");
    m->p.audio_vocab = (int32_t)meta_u32(g, "omnivoice.audio_vocab_size");
    m->p.audio_mask_id = (int32_t)meta_u32(g, "omnivoice.audio_mask_id");
    m->p.n_codebook = (int32_t)meta_u32(g, "omnivoice.num_audio_codebook");

    /* backend */
    if (use_gpu) {
#ifdef GGML_USE_CUDA
        m->backend = ggml_backend_cuda_init(0);
#elif defined(GGML_USE_VULKAN)
        m->backend = ggml_backend_vk_init(0);
#endif
        if (!m->backend) {
            fprintf(stderr, "ov_model: backend de GPU indisponivel\n");
            gguf_free(g);
            ov_model_free(m);
            return NULL;
        }
    } else {
        m->backend = ggml_backend_cpu_init();
    }

    /* aloca os tensores de pesos no backend e carrega do arquivo */
    m->weights_buf = ggml_backend_alloc_ctx_tensors(meta_ctx, m->backend);
    if (!m->weights_buf) {
        fprintf(stderr, "ov_model: sem memoria para os pesos\n");
        gguf_free(g);
        ov_model_free(m);
        return NULL;
    }

    {
        FILE *f = fopen(gguf_path, "rb");
        if (!f) {
            gguf_free(g);
            ov_model_free(m);
            return NULL;
        }
        const size_t data_off = gguf_get_data_offset(g);
        const int64_t n_tensors = gguf_get_n_tensors(g);
        size_t max_sz = 0;
        for (int64_t i = 0; i < n_tensors; i++) {
            struct ggml_tensor *t =
                ggml_get_tensor(meta_ctx, gguf_get_tensor_name(g, i));
            size_t sz = ggml_nbytes(t);
            if (sz > max_sz) max_sz = sz;
        }
        void *staging = malloc(max_sz);
        for (int64_t i = 0; i < n_tensors; i++) {
            const char *name = gguf_get_tensor_name(g, i);
            struct ggml_tensor *t = ggml_get_tensor(meta_ctx, name);
            size_t sz = ggml_nbytes(t);
            if (ov_file_seek(
                    f, (int64_t)(data_off + gguf_get_tensor_offset(g, i))) !=
                0) {
                fprintf(stderr, "ov_model: falha ao buscar tensor %s\n",
                        name);
                free(staging);
                fclose(f);
                gguf_free(g);
                ov_model_free(m);
                return NULL;
            }
            if (fread(staging, 1, sz, f) != sz) {
                fprintf(stderr, "ov_model: leitura truncada em %s\n", name);
                free(staging);
                fclose(f);
                gguf_free(g);
                ov_model_free(m);
                return NULL;
            }
            ggml_backend_tensor_set(t, staging, 0, sz);
        }
        free(staging);
        fclose(f);
    }
    gguf_free(g);

    /* resolve os tensores por nome */
    m->token_embd = get_weight(meta_ctx, "token_embd.weight");
    m->audio_embd = get_weight(meta_ctx, "audio_embd.weight");
    m->audio_heads = get_weight(meta_ctx, "audio_heads.weight");
    m->output_norm = get_weight(meta_ctx, "output_norm.weight");
    if (!m->token_embd || !m->audio_embd || !m->audio_heads ||
        !m->output_norm) {
        ov_model_free(m);
        return NULL;
    }

    m->layers = calloc(m->p.n_layer, sizeof(ov_layer));
    for (int i = 0; i < m->p.n_layer; i++) {
        char name[128];
        ov_layer *l = &m->layers[i];
#define L(field, suffix)                                        \
    snprintf(name, sizeof(name), "blk.%d." suffix, i);          \
    l->field = get_weight(meta_ctx, name);                      \
    if (!l->field) {                                            \
        ov_model_free(m);                                       \
        return NULL;                                            \
    }
        L(attn_norm, "attn_norm.weight")
        L(wq, "attn_q.weight")
        L(wk, "attn_k.weight")
        L(wv, "attn_v.weight")
        L(wo, "attn_output.weight")
        L(q_norm, "attn_q_norm.weight")
        L(k_norm, "attn_k_norm.weight")
        L(ffn_norm, "ffn_norm.weight")
        L(w_gate, "ffn_gate.weight")
        L(w_up, "ffn_up.weight")
        L(w_down, "ffn_down.weight")
#undef L
    }

    /* token_remap para o host */
    {
        struct ggml_tensor *t = get_weight(meta_ctx, "token_remap");
        if (!t) {
            ov_model_free(m);
            return NULL;
        }
        m->remap = malloc((size_t)m->p.vocab_full * sizeof(int32_t));
        ggml_backend_tensor_get(t, m->remap, 0,
                                (size_t)m->p.vocab_full * sizeof(int32_t));
    }

    m->n_plans = 0;

    fprintf(stderr, "ov_model: pesos em %.0f MB (%s)\n",
            (double)ggml_backend_buffer_get_size(m->weights_buf) / 1e6,
            ggml_backend_name(m->backend));

    return m;
}

static void ov_plan_free(ov_fwd_plan *pl) {
    if (pl->galloc) ggml_gallocr_free(pl->galloc);
    if (pl->ctx) ggml_free(pl->ctx);
    memset(pl, 0, sizeof(*pl));
}

void ov_model_free(ov_model *m) {
    if (!m) return;
    for (int i = 0; i < m->n_plans; i++) ov_plan_free(&m->plans[i]);
    if (m->weights_buf) ggml_backend_buffer_free(m->weights_buf);
    if (m->backend) ggml_backend_free(m->backend);
    if (m->weights_ctx) ggml_free(m->weights_ctx);
    free(m->layers);
    free(m->remap);
    free(m);
}

void ov_model_set_threads(ov_model *m, int n_threads) {
    if (m && m->backend && ggml_backend_is_cpu(m->backend)) {
        ggml_backend_cpu_set_n_threads(m->backend, n_threads);
    }
}

const ov_model_params *ov_model_get_params(const ov_model *m) { return &m->p; }

size_t ov_model_weights_bytes(const ov_model *m) {
    return m->weights_buf ? ggml_backend_buffer_get_size(m->weights_buf) : 0;
}

int32_t ov_model_remap_token(const ov_model *m, int32_t token_id) {
    if (token_id < 0 || token_id >= m->p.vocab_full) return -1;
    return m->remap[token_id];
}

/* ------------------------------------------------------------------ */
/* Forward                                                             */
/* ------------------------------------------------------------------ */

/* Constrói o grafo persistente para uma forma (seq, n_tail) e o aloca uma
 * única vez. Depois disso, cada passo só reescreve o conteúdo das entradas. */
static int build_plan(ov_model *m, ov_fwd_plan *pl, int32_t seq,
                      int32_t n_tail) {
    const ov_model_params *p = &m->p;
    const int C = p->n_codebook;

    /* contexto do grafo (apenas metadados; dados via gallocr) */
    const size_t n_nodes_max = (size_t)(128 + 48 * p->n_layer);
    size_t ctx_size = ggml_tensor_overhead() * n_nodes_max +
                      ggml_graph_overhead_custom(n_nodes_max, false);
    struct ggml_init_params ip = {
        /*mem_size  =*/ctx_size,
        /*mem_buffer=*/NULL,
        /*no_alloc  =*/true,
    };
    struct ggml_context *ctx = ggml_init(ip);

    /* --- entradas --- */
    struct ggml_tensor *text_ids =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_set_name(text_ids, "text_ids");
    ggml_set_input(text_ids);

    struct ggml_tensor *audio_ids =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)C * seq);
    ggml_set_name(audio_ids, "audio_ids");
    ggml_set_input(audio_ids);

    struct ggml_tensor *amask =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, seq);
    ggml_set_name(amask, "amask");
    ggml_set_input(amask);

    struct ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    /* --- embeddings --- */
    struct ggml_tensor *text_embd =
        ggml_get_rows(ctx, m->token_embd, text_ids); /* [1024, seq] */

    struct ggml_tensor *audio_rows =
        ggml_get_rows(ctx, m->audio_embd, audio_ids); /* [1024, C*seq] */
    struct ggml_tensor *audio_3d =
        ggml_reshape_3d(ctx, audio_rows, p->n_embd, seq, C);
    /* soma dos C codebooks: [1024, seq, C] -> soma em dim2 */
    struct ggml_tensor *audio_sum = NULL;
    for (int cb = 0; cb < C; cb++) {
        struct ggml_tensor *slice = ggml_view_2d(
            ctx, audio_3d, p->n_embd, seq, audio_3d->nb[1],
            (size_t)cb * audio_3d->nb[2]);
        audio_sum = audio_sum ? ggml_add(ctx, audio_sum, slice) : slice;
    }

    /* seleção: x = text*(1-mask) + audio*mask */
    struct ggml_tensor *one_minus =
        ggml_scale_bias(ctx, amask, -1.0f, 1.0f); /* 1 - mask */
    struct ggml_tensor *x =
        ggml_add(ctx, ggml_mul(ctx, text_embd, one_minus),
                 ggml_mul(ctx, audio_sum, amask));

    const float kq_scale = 1.0f / sqrtf((float)p->head_dim);

    for (int il = 0; il < p->n_layer; il++) {
        ov_layer *l = &m->layers[il];

        /* --- atenção --- */
        struct ggml_tensor *cur = ggml_rms_norm(ctx, x, p->rms_norm_eps);
        cur = ggml_mul(ctx, cur, l->attn_norm);

        struct ggml_tensor *q = ggml_mul_mat(ctx, l->wq, cur);
        struct ggml_tensor *k = ggml_mul_mat(ctx, l->wk, cur);
        struct ggml_tensor *v = ggml_mul_mat(ctx, l->wv, cur);

        q = ggml_reshape_3d(ctx, q, p->head_dim, p->n_head, seq);
        k = ggml_reshape_3d(ctx, k, p->head_dim, p->n_head_kv, seq);
        v = ggml_reshape_3d(ctx, v, p->head_dim, p->n_head_kv, seq);

        /* RMSNorm por cabeça (Qwen3) */
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, p->rms_norm_eps), l->q_norm);
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, p->rms_norm_eps), l->k_norm);

        q = ggml_rope_ext(ctx, q, pos, NULL, p->head_dim, GGML_ROPE_TYPE_NEOX,
                          0, p->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, NULL, p->head_dim, GGML_ROPE_TYPE_NEOX,
                          0, p->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        /* [hd, H, S] -> [hd, S, H] */
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

        /* atenção completa, sem máscara causal */
        struct ggml_tensor *kq = ggml_mul_mat(ctx, k, q); /* [S, S, H] */
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, NULL, kq_scale, 0.0f);

        /* v: [hd, KVH, S] -> [S, hd, KVH] */
        struct ggml_tensor *vt =
            ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        struct ggml_tensor *kqv = ggml_mul_mat(ctx, vt, kq); /* [hd, S, H] */
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        kqv = ggml_reshape_2d(ctx, kqv, (int64_t)p->head_dim * p->n_head, seq);

        cur = ggml_mul_mat(ctx, l->wo, kqv);
        x = ggml_add(ctx, x, cur);

        /* --- FFN (SwiGLU) --- */
        cur = ggml_rms_norm(ctx, x, p->rms_norm_eps);
        cur = ggml_mul(ctx, cur, l->ffn_norm);
        struct ggml_tensor *gate = ggml_mul_mat(ctx, l->w_gate, cur);
        struct ggml_tensor *up = ggml_mul_mat(ctx, l->w_up, cur);
        cur = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        cur = ggml_mul_mat(ctx, l->w_down, cur);
        x = ggml_add(ctx, x, cur);
    }

    x = ggml_rms_norm(ctx, x, p->rms_norm_eps);
    x = ggml_mul(ctx, x, m->output_norm);

    /* apenas as últimas n_tail posições passam pelo audio_heads */
    struct ggml_tensor *tail = ggml_view_2d(
        ctx, x, p->n_embd, n_tail, x->nb[1], (size_t)(seq - n_tail) * x->nb[1]);
    struct ggml_tensor *logits =
        ggml_mul_mat(ctx, m->audio_heads, ggml_cont(ctx, tail));
    /* [C*audio_vocab, n_tail] */
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, n_nodes_max, false);
    ggml_build_forward_expand(gf, logits);

    pl->galloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ggml_gallocr_alloc_graph(pl->galloc, gf)) {
        fprintf(stderr, "ov_model: falha ao alocar grafo (seq=%d)\n", seq);
        ggml_gallocr_free(pl->galloc);
        ggml_free(ctx);
        pl->galloc = NULL;
        return -1;
    }

    fprintf(stderr, "ov_model: grafo (seq=%d, n_tail=%d) alocado, %.0f MB\n",
            seq, n_tail, (double)ggml_gallocr_get_buffer_size(pl->galloc, 0) / 1e6);

    pl->ctx = ctx;
    pl->gf = gf;
    pl->seq = seq;
    pl->n_tail = n_tail;
    pl->text_ids = text_ids;
    pl->audio_ids = audio_ids;
    pl->amask = amask;
    pl->pos = pos;
    pl->logits = logits;
    pl->valid = 1;
    return 0;
}

/* Retorna um plano para (seq, n_tail), construindo/reaproveitando o cache. */
static ov_fwd_plan *get_plan(ov_model *m, int32_t seq, int32_t n_tail) {
    for (int i = 0; i < m->n_plans; i++) {
        if (m->plans[i].valid && m->plans[i].seq == seq &&
            m->plans[i].n_tail == n_tail) {
            return &m->plans[i];
        }
    }
    /* miss: usa slot livre ou evita o mais antigo (formas mudam entre chunks) */
    ov_fwd_plan *pl;
    if (m->n_plans < OV_MAX_PLANS) {
        pl = &m->plans[m->n_plans++];
    } else {
        /* evita o slot 0 e rotaciona (raro: só quando o comprimento muda) */
        ov_plan_free(&m->plans[0]);
        m->plans[0] = m->plans[1];
        pl = &m->plans[1];
        memset(pl, 0, sizeof(*pl));
    }
    if (build_plan(m, pl, seq, n_tail) != 0) {
        pl->valid = 0;
        return NULL;
    }
    return pl;
}

int ov_model_forward(ov_model *m, const int32_t *input_ids,
                     const uint8_t *audio_mask, int32_t seq, int32_t n_tail,
                     float *out_logits) {
    const ov_model_params *p = &m->p;
    const int C = p->n_codebook;

    ov_fwd_plan *pl = get_plan(m, seq, n_tail);
    if (!pl) return -1;

    /* --- preenche as entradas (mesmos tensores a cada passo) --- */
    {
        int32_t *tids = malloc((size_t)seq * sizeof(int32_t));
        int32_t *aids = malloc((size_t)C * seq * sizeof(int32_t));
        float *mk = malloc((size_t)seq * sizeof(float));
        int32_t *pp = malloc((size_t)seq * sizeof(int32_t));

        for (int32_t s = 0; s < seq; s++) {
            mk[s] = audio_mask[s] ? 1.0f : 0.0f;
            pp[s] = s;
            if (audio_mask[s]) {
                tids[s] = 0; /* descartado pela seleção */
            } else {
                tids[s] = input_ids[s]; /* linha 0, já remapeado */
            }
            for (int cb = 0; cb < C; cb++) {
                int32_t id = input_ids[(size_t)cb * seq + s];
                /* offset por codebook; posições de texto: 0 (descartado) */
                aids[(size_t)cb * seq + s] =
                    audio_mask[s] ? id + cb * p->audio_vocab : 0;
            }
        }

        ggml_backend_tensor_set(pl->text_ids, tids, 0,
                                (size_t)seq * sizeof(int32_t));
        ggml_backend_tensor_set(pl->audio_ids, aids, 0,
                                (size_t)C * seq * sizeof(int32_t));
        ggml_backend_tensor_set(pl->amask, mk, 0, (size_t)seq * sizeof(float));
        ggml_backend_tensor_set(pl->pos, pp, 0, (size_t)seq * sizeof(int32_t));

        free(tids);
        free(aids);
        free(mk);
        free(pp);
    }

    if (ggml_backend_graph_compute(m->backend, pl->gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ov_model: falha no compute\n");
        return -1;
    }

    /* --- lê e reordena logits: [C*V, n_tail] -> [C][pos][V] --- */
    {
        const int64_t V = p->audio_vocab;
        const size_t row = (size_t)C * V;
        float *raw = malloc(row * (size_t)n_tail * sizeof(float));
        ggml_backend_tensor_get(pl->logits, raw, 0,
                                row * (size_t)n_tail * sizeof(float));
        for (int cb = 0; cb < C; cb++) {
            for (int32_t t = 0; t < n_tail; t++) {
                memcpy(out_logits + ((size_t)cb * n_tail + t) * V,
                       raw + (size_t)t * row + (size_t)cb * V,
                       (size_t)V * sizeof(float));
            }
        }
        free(raw);
    }

    return 0;
}
