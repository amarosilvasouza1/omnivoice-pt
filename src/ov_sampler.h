/* Sampler de difusão mascarada do OmniVoice (port de _generate_iterative).
 *
 * Diferenças deliberadas em relação ao pipeline PyTorch:
 *   - batch 1, cond e uncond como forwards separados com o comprimento
 *     real de cada um (sem padding desperdiçado);
 *   - CFG fundido no espaço de log-probs, calculado só nas posições alvo.
 */

#ifndef OV_SAMPLER_H
#define OV_SAMPLER_H

#include <stdint.h>

#include "ov_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t num_step;            /* 32 */
    float guidance_scale;        /* 2.0 */
    float t_shift;               /* 0.1 */
    float layer_penalty_factor;  /* 5.0 */
    float position_temperature;  /* 5.0 */
    float class_temperature;     /* 0.0 = greedy */
    uint64_t seed;
} ov_sample_config;

ov_sample_config ov_sample_config_default(void);

/* Gera os tokens de áudio.
 *
 * prefix_ids:  i32[n_codebook * prefix_len] — estilo + texto (+ tokens de
 *              áudio de referência). Posições de texto: ids remapeados na
 *              linha 0; posições de áudio de referência: ids 0..1024 em
 *              todas as linhas.
 * prefix_mask: u8[prefix_len] — 1 nas posições de áudio de referência.
 * target_len:  número de frames de áudio a gerar.
 * out_tokens:  i32[n_codebook * target_len].
 * Retorna 0 em sucesso. */
int ov_sample(ov_model *m, const int32_t *prefix_ids,
              const uint8_t *prefix_mask, int32_t prefix_len,
              int32_t target_len, const ov_sample_config *cfg,
              int32_t *out_tokens);

#ifdef __cplusplus
}
#endif

#endif /* OV_SAMPLER_H */
