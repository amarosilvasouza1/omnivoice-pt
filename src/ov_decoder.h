/* Decoder de áudio (caminho de decode do HiggsAudioV2Tokenizer):
 *   RVQ (8 codebooks) -> fc2 -> DacDecoder -> PCM 24 kHz.
 * Roda em CPU (é pequeno: ~22M params) via GGML.
 */

#ifndef OV_DECODER_H
#define OV_DECODER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ov_decoder ov_decoder;

/* use_gpu: 1 tenta CUDA (cai para CPU se indisponível), 0 força CPU. */
ov_decoder *ov_decoder_load(const char *gguf_path, int use_gpu);
void ov_decoder_set_threads(ov_decoder *d, int n_threads);
void ov_decoder_free(ov_decoder *d);

int32_t ov_decoder_sampling_rate(const ov_decoder *d);
int32_t ov_decoder_hop_length(const ov_decoder *d);
size_t ov_decoder_weights_bytes(const ov_decoder *d);

/* Decodifica tokens (n_codebook x n_frames, row-major por codebook) para
 * PCM float. Escreve n_frames*hop_length amostras em out_audio.
 * Retorna número de amostras, ou -1 em erro. */
int64_t ov_decoder_decode(ov_decoder *d, const int32_t *codes,
                          int32_t n_frames, float *out_audio);

#ifdef __cplusplus
}
#endif

#endif /* OV_DECODER_H */
