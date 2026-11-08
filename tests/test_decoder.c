/* Valida o decoder C contra o golden do PyTorch (decoder_golden.bin).
 *
 * Uso: test_decoder <higgs-decoder.gguf> <decoder_golden.bin>
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ov_decoder.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: %s decoder.gguf decoder_golden.bin [gpu]\n",
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
    uint32_t C, T, N;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "OVDG", 4) != 0 ||
        fread(&C, 4, 1, f) != 1 || fread(&T, 4, 1, f) != 1 ||
        fread(&N, 4, 1, f) != 1) {
        fprintf(stderr, "golden invalido\n");
        return 2;
    }
    int32_t *codes = malloc((size_t)C * T * 4);
    float *want = malloc((size_t)N * 4);
    if (fread(codes, 4, (size_t)C * T, f) != (size_t)C * T ||
        fread(want, 4, N, f) != N) {
        fprintf(stderr, "golden truncado\n");
        return 2;
    }
    fclose(f);
    printf("golden: %u codebooks x %u frames -> %u amostras\n", C, T, N);

    ov_decoder *d = ov_decoder_load(argv[1], use_gpu);
    if (!d) return 2;
    printf("decoder: %d Hz, hop %d (%s)\n", ov_decoder_sampling_rate(d),
           ov_decoder_hop_length(d), use_gpu ? "GPU" : "CPU");

    float *got = malloc((size_t)N * 4);
    int64_t n = ov_decoder_decode(d, codes, (int32_t)T, got);
    if (n < 0) return 1;
    printf("decodificadas %lld amostras\n", (long long)n);

    double dot = 0, na = 0, nb = 0, max_abs = 0;
    for (int64_t i = 0; i < n; i++) {
        double a = want[i], b = got[i];
        dot += a * b;
        na += a * a;
        nb += b * b;
        double diff = fabs(a - b);
        if (diff > max_abs) max_abs = diff;
    }
    double cos_sim = dot / (sqrt(na) * sqrt(nb) + 1e-12);
    double rms = sqrt(na / (double)n);

    printf("cosine similarity : %.6f\n", cos_sim);
    printf("max abs diff      : %.5f (rms do sinal: %.4f)\n", max_abs, rms);

    ov_decoder_free(d);
    free(codes);
    free(want);
    free(got);

    if (cos_sim < 0.999 || (int64_t)N != n) {
        printf("FALHOU\n");
        return 1;
    }
    printf("PASSOU\n");
    return 0;
}
