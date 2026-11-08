/* Modelo OmniVoice (backbone Qwen3 + audio embeddings/heads) sobre GGML.
 *
 * Carrega o GGUF gerado por tools/export.py e executa o forward
 * bidirecional usado pela difusão mascarada. Sem KV cache (cada passo
 * reprocessa a sequência, como no pipeline original), sem padding:
 * cond e uncond são grafos independentes com o comprimento real.
 */

#ifndef OV_MODEL_H
#define OV_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ov_model ov_model;

typedef struct {
    int32_t n_layer;
    int32_t n_embd;
    int32_t n_head;
    int32_t n_head_kv;
    int32_t head_dim;
    int32_t n_ff;
    float rms_norm_eps;
    float rope_theta;
    int32_t vocab_full;   /* 151676: ids originais do tokenizer */
    int32_t vocab_kept;   /* linhas da matriz de embeddings podada */
    int32_t audio_vocab;  /* 1025 */
    int32_t audio_mask_id; /* 1024 */
    int32_t n_codebook;   /* 8 */
} ov_model_params;

/* use_gpu: 1 tenta CUDA (falha -> erro), 0 CPU. */
ov_model *ov_model_load(const char *gguf_path, int use_gpu);
void ov_model_set_threads(ov_model *m, int n_threads);

void ov_model_free(ov_model *m);

const ov_model_params *ov_model_get_params(const ov_model *m);

/* Remapeia um id de texto original para a linha podada da matriz de
 * embeddings. Ids podados caem em -1 (o chamador deve então re-tokenizar
 * via bytes — com o keep-set atual isso não ocorre para texto PT). */
int32_t ov_model_remap_token(const ov_model *m, int32_t token_id);

/* Forward completo (bidirecional, sem máscara causal).
 *
 * input_ids:  i32[n_codebook * seq]  row-major (codebook, seq).
 *             Linha 0 carrega ids de texto nas posições de texto
 *             (JÁ REMAPEADOS para o vocabulário podado).
 *             Posições de áudio carregam ids de áudio 0..1024 em todas
 *             as linhas (sem offset de codebook; o offset é interno).
 * audio_mask: u8[seq], 1 = posição de áudio.
 * out_logits: f32[n_codebook * n_tail * audio_vocab] para as últimas
 *             n_tail posições (ordem: codebook, posição, vocab).
 * Retorna 0 em sucesso. */
int ov_model_forward(ov_model *m, const int32_t *input_ids,
                     const uint8_t *audio_mask, int32_t seq, int32_t n_tail,
                     float *out_logits);

/* Bytes de VRAM/RAM ocupados pelos pesos do modelo (para relatórios). */
size_t ov_model_weights_bytes(const ov_model *m);

#ifdef __cplusplus
}
#endif

#endif /* OV_MODEL_H */
