/* Valida o tokenizer C contra os vetores dourados do tokenizer HF.
 *
 * Uso: test_tokenizer <tokenizer.bin> <tokenizer_golden.bin>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ov_tokenizer.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "uso: %s tokenizer.bin tokenizer_golden.bin\n",
                argv[0]);
        return 2;
    }

    ov_tokenizer *tk = ov_tokenizer_load(argv[1]);
    if (!tk) return 2;
    printf("tokenizer: %d tokens no vocabulario\n",
           ov_tokenizer_vocab_size(tk));

    /* sanity: tokens especiais resolvem */
    const char *specials[] = {"<|denoise|>",       "<|lang_start|>",
                              "<|lang_end|>",      "<|instruct_start|>",
                              "<|instruct_end|>",  "<|text_start|>",
                              "<|text_end|>"};
    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        int32_t id = ov_tokenizer_special_id(tk, specials[i]);
        printf("  %s -> %d\n", specials[i], id);
        if (id < 0) {
            fprintf(stderr, "FALHA: token especial ausente\n");
            return 1;
        }
    }

    FILE *f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "nao abriu %s\n", argv[2]);
        return 2;
    }
    char magic[4];
    uint32_t n_cases;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "OVGT", 4) != 0 ||
        fread(&n_cases, 4, 1, f) != 1) {
        fprintf(stderr, "golden invalido\n");
        return 2;
    }

    int failures = 0;
    for (uint32_t c = 0; c < n_cases; c++) {
        uint32_t text_len, n_ids;
        if (fread(&text_len, 4, 1, f) != 1) return 2;
        char *text = malloc((size_t)text_len + 1);
        if (fread(text, 1, text_len, f) != text_len) return 2;
        text[text_len] = 0;
        if (fread(&n_ids, 4, 1, f) != 1) return 2;
        uint32_t *want = malloc((size_t)n_ids * 4);
        if (fread(want, 4, n_ids, f) != n_ids) return 2;

        int32_t got[4096];
        int32_t n_got = ov_tokenizer_encode(tk, text, got, 4096);

        int ok = (n_got == (int32_t)n_ids);
        if (ok) {
            for (uint32_t i = 0; i < n_ids; i++) {
                if (got[i] != (int32_t)want[i]) {
                    ok = 0;
                    break;
                }
            }
        }

        if (ok) {
            printf("OK   [%u] %.50s\n", c, text);
        } else {
            failures++;
            printf("FAIL [%u] %.50s\n", c, text);
            printf("  esperado (%u):", n_ids);
            for (uint32_t i = 0; i < n_ids && i < 40; i++) {
                printf(" %u", want[i]);
            }
            printf("\n  obtido   (%d):", n_got);
            for (int32_t i = 0; i < n_got && i < 40; i++) {
                printf(" %d", got[i]);
            }
            printf("\n");
        }
        free(text);
        free(want);
    }
    fclose(f);
    ov_tokenizer_free(tk);

    if (failures) {
        printf("\n%d caso(s) FALHARAM\n", failures);
        return 1;
    }
    printf("\ntodos os casos passaram\n");
    return 0;
}
