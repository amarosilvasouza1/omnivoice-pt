/* Valida o forward GGML contra o golden do PyTorch (model_golden.bin).
 *
 * Uso: test_model <modelo.gguf> <model_golden.bin> [gpu]
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ov_model.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: %s modelo.gguf model_golden.bin [gpu]\n",
                argv[0]);
        return 2;
    }
    int use_gpu = (argc > 3 && strcmp(argv[3], "gpu") == 0);

    FILE *f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "nao abriu %s\n", argv[2]);
        return 2;
    }
    char magic[4];
    uint32_t C, S, n_tail, V;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "OVMG", 4) != 0 ||
        fread(&C, 4, 1, f) != 1 || fread(&S, 4, 1, f) != 1 ||
        fread(&n_tail, 4, 1, f) != 1 || fread(&V, 4, 1, f) != 1) {
        fprintf(stderr, "golden invalido\n");
        return 2;
    }
    printf("golden: C=%u S=%u n_tail=%u V=%u\n", C, S, n_tail, V);

    int32_t *ids = malloc((size_t)C * S * 4);
    uint8_t *mask = malloc(S);
    size_t n_logit = (size_t)C * n_tail * V;
    float *want = malloc(n_logit * 4);
    if (fread(ids, 4, (size_t)C * S, f) != (size_t)C * S ||
        fread(mask, 1, S, f) != S ||
        fread(want, 4, n_logit, f) != n_logit) {
        fprintf(stderr, "golden truncado\n");
        return 2;
    }
    fclose(f);

    ov_model *m = ov_model_load(argv[1], use_gpu);
    if (!m) return 2;
    const ov_model_params *p = ov_model_get_params(m);
    printf("modelo: %d camadas, embd %d, vocab podado %d/%d (%s)\n",
           p->n_layer, p->n_embd, p->vocab_kept, p->vocab_full,
           use_gpu ? "GPU" : "CPU");

    /* remapeia ids de texto (linha 0; posições de áudio ficam) */
    for (uint32_t s = 0; s < S; s++) {
        if (!mask[s]) {
            int32_t orig = ids[s];
            int32_t re = ov_model_remap_token(m, orig);
            if (re < 0) {
                fprintf(stderr, "token %d fora do keep-set!\n", orig);
                return 1;
            }
            ids[s] = re;
        }
    }

    float *got = malloc(n_logit * 4);
    if (ov_model_forward(m, ids, mask, (int32_t)S, (int32_t)n_tail, got) !=
        0) {
        return 1;
    }

    /* métricas */
    double dot = 0, na = 0, nb = 0, max_abs = 0, sum_abs = 0;
    for (size_t i = 0; i < n_logit; i++) {
        double a = want[i], b = got[i];
        dot += a * b;
        na += a * a;
        nb += b * b;
        double d = fabs(a - b);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
    }
    double cos_sim = dot / (sqrt(na) * sqrt(nb) + 1e-12);

    int argmax_ok = 0, argmax_total = 0;
    for (uint32_t cb = 0; cb < C; cb++) {
        for (uint32_t t = 0; t < n_tail; t++) {
            const float *wrow = want + ((size_t)cb * n_tail + t) * V;
            const float *grow = got + ((size_t)cb * n_tail + t) * V;
            uint32_t wa = 0, ga = 0;
            for (uint32_t v = 1; v < V; v++) {
                if (wrow[v] > wrow[wa]) wa = v;
                if (grow[v] > grow[ga]) ga = v;
            }
            argmax_total++;
            if (wa == ga) argmax_ok++;
        }
    }

    printf("cosine similarity : %.6f\n", cos_sim);
    printf("max abs diff      : %.4f\n", max_abs);
    printf("mean abs diff     : %.4f\n", sum_abs / (double)n_logit);
    printf("argmax match      : %d/%d\n", argmax_ok, argmax_total);

    ov_model_free(m);
    free(ids);
    free(mask);
    free(want);
    free(got);

    /* Critérios: a estrutura tem de bater (cosseno); o argmax neste golden
     * é instável por construção (posições 100% mascaradas = entropia máxima,
     * f16 da GPU flipa empates). O gate decisivo é o A/B de áudio da fase 4. */
    if (cos_sim < 0.9999 || argmax_ok < argmax_total * 7 / 10) {
        printf("FALHOU\n");
        return 1;
    }
    printf("PASSOU\n");
    return 0;
}
