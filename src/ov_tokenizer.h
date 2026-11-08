/* Tokenizer BPE byte-level (Qwen3) para o runtime OmniVoice-PT.
 *
 * Carrega o formato compacto tokenizer.bin (OVTK) gerado por
 * tools/export.py e reproduz exatamente a tokenização do HuggingFace:
 * pre-tokenização com o regex do Qwen2 + BPE greedy por rank de merge.
 *
 * A tokenização usa sempre os IDs ORIGINAIS do vocabulário completo;
 * o remapeamento para as linhas podadas da matriz de embeddings é feito
 * pelo modelo (tensor token_remap do GGUF), não aqui.
 */

#ifndef OV_TOKENIZER_H
#define OV_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ov_tokenizer ov_tokenizer;

/* Carrega tokenizer.bin. Retorna NULL em caso de erro. */
ov_tokenizer *ov_tokenizer_load(const char *path);

void ov_tokenizer_free(ov_tokenizer *tk);

/* Tokeniza texto UTF-8 (assumido NFC, sem tokens especiais embutidos).
 * Escreve até max_ids em out_ids e retorna o número de tokens,
 * ou -1 se estourar max_ids. */
int32_t ov_tokenizer_encode(const ov_tokenizer *tk, const char *text,
                            int32_t *out_ids, int32_t max_ids);

/* Resolve o ID de um token especial pelo seu texto literal
 * (ex.: "<|denoise|>"). Retorna -1 se não existir. */
int32_t ov_tokenizer_special_id(const ov_tokenizer *tk, const char *literal);

/* Número total de tokens no vocabulário. */
int32_t ov_tokenizer_vocab_size(const ov_tokenizer *tk);

#ifdef __cplusplus
}
#endif

#endif /* OV_TOKENIZER_H */
