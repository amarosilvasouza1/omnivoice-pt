/* Implementação do decoder de áudio (ver ov_decoder.h). */

#include "ov_decoder.h"

#include <stdarg.h>
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

#define N_CODEBOOK 8
#define N_BLOCK 5

/* Limita o maior grafo temporario do decoder na GPU. Os doze frames de
 * contexto (480 ms de cada lado) preservam a continuidade nas emendas. O
 * custo extra fica concentrado no decoder, uma pequena parte do tempo total. */
#define OV_DECODE_CHUNK_FRAMES 512
#define OV_DECODE_CONTEXT_FRAMES 12

typedef struct {
    struct ggml_tensor *embed;    /* [64, 1024] (ne) */
    struct ggml_tensor *proj_w;   /* [64, 1024] */
    struct ggml_tensor *proj_b;   /* [1024] */
} rvq_level;

typedef struct {
    struct ggml_tensor *snake_alpha; /* pré-bloco */
    struct ggml_tensor *convt_w, *convt_b;
    int stride, pad, out_pad, kernel;
    /* 3 res units: convs k7 (d=1,3,9) + k1 */
    struct ggml_tensor *ru_alpha1[3], *ru_conv1_w[3], *ru_conv1_b[3];
    struct ggml_tensor *ru_alpha2[3], *ru_conv2_w[3], *ru_conv2_b[3];
} dac_block;

struct ov_decoder {
    ggml_backend_t backend;
    ggml_backend_buffer_t buf;
    struct ggml_context *wctx;
    ggml_gallocr_t galloc;
    int is_gpu;

    int32_t sampling_rate, hop_length;

    rvq_level rvq[N_CODEBOOK];
    struct ggml_tensor *fc2_w, *fc2_b;
    struct ggml_tensor *conv1_w, *conv1_b;
    dac_block blocks[N_BLOCK];
    struct ggml_tensor *final_alpha, *final_conv_w, *final_conv_b;
};

static const int BLOCK_STRIDE[N_BLOCK] = {8, 5, 4, 2, 3};
static const int BLOCK_KERNEL[N_BLOCK] = {16, 10, 8, 4, 6};
static const int BLOCK_PAD[N_BLOCK] = {4, 3, 2, 1, 2};
static const int BLOCK_OUTPAD[N_BLOCK] = {0, 1, 0, 0, 1};
static const int RU_DILATION[3] = {1, 3, 9};
static const int RU_PAD[3] = {3, 9, 27};

static struct ggml_tensor *W(struct ggml_context *ctx, const char *fmt, ...) {
    char name[192];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    struct ggml_tensor *t = ggml_get_tensor(ctx, name);
    if (!t) fprintf(stderr, "ov_decoder: tensor ausente: %s\n", name);
    return t;
}

ov_decoder *ov_decoder_load(const char *gguf_path, int use_gpu) {
    struct ggml_context *meta = NULL;
    struct gguf_init_params gp = {/*no_alloc=*/true, /*ctx=*/&meta};
    struct gguf_context *g = gguf_init_from_file(gguf_path, gp);
    if (!g) {
        fprintf(stderr, "ov_decoder: falha ao abrir %s\n", gguf_path);
        return NULL;
    }

    ov_decoder *d = calloc(1, sizeof(*d));
    d->wctx = meta;
    d->backend = NULL;
    if (use_gpu) {
#ifdef GGML_USE_CUDA
        d->backend = ggml_backend_cuda_init(0);
        if (d->backend) d->is_gpu = 1;
#elif defined(GGML_USE_VULKAN)
        d->backend = ggml_backend_vk_init(0);
        if (d->backend) d->is_gpu = 1;
#endif
        if (!d->backend) {
            fprintf(stderr, "ov_decoder: backend de GPU indisponivel, usando CPU\n");
        }
    }
    if (!d->backend) d->backend = ggml_backend_cpu_init();

    {
        int64_t i = gguf_find_key(g, "higgs.sampling_rate");
        d->sampling_rate = i >= 0 ? (int32_t)gguf_get_val_u32(g, i) : 24000;
        i = gguf_find_key(g, "higgs.hop_length");
        d->hop_length = i >= 0 ? (int32_t)gguf_get_val_u32(g, i) : 960;
    }

    d->buf = ggml_backend_alloc_ctx_tensors(meta, d->backend);
    {
        FILE *f = fopen(gguf_path, "rb");
        if (!f) {
            gguf_free(g);
            ov_decoder_free(d);
            return NULL;
        }
        const size_t off = gguf_get_data_offset(g);
        const int64_t n = gguf_get_n_tensors(g);
        for (int64_t i = 0; i < n; i++) {
            const char *name = gguf_get_tensor_name(g, i);
            struct ggml_tensor *t = ggml_get_tensor(meta, name);
            size_t sz = ggml_nbytes(t);
            void *tmp = malloc(sz);
            if (ov_file_seek(
                    f, (int64_t)(off + gguf_get_tensor_offset(g, i))) != 0) {
                fprintf(stderr, "ov_decoder: falha ao buscar tensor %s\n",
                        name);
                free(tmp);
                fclose(f);
                gguf_free(g);
                ov_decoder_free(d);
                return NULL;
            }
            if (fread(tmp, 1, sz, f) != sz) {
                fprintf(stderr, "ov_decoder: leitura truncada: %s\n", name);
                free(tmp);
                fclose(f);
                gguf_free(g);
                ov_decoder_free(d);
                return NULL;
            }
            ggml_backend_tensor_set(t, tmp, 0, sz);
            free(tmp);
        }
        fclose(f);
    }
    gguf_free(g);

    int ok = 1;
#define REQ(dst, ...)                     \
    do {                                  \
        (dst) = W(meta, __VA_ARGS__);     \
        if (!(dst)) ok = 0;               \
    } while (0)

    for (int cb = 0; cb < N_CODEBOOK; cb++) {
        REQ(d->rvq[cb].embed, "quantizer.quantizers.%d.codebook.embed", cb);
        REQ(d->rvq[cb].proj_w, "quantizer.quantizers.%d.project_out.weight",
            cb);
        REQ(d->rvq[cb].proj_b, "quantizer.quantizers.%d.project_out.bias", cb);
    }
    REQ(d->fc2_w, "fc2.weight");
    REQ(d->fc2_b, "fc2.bias");
    REQ(d->conv1_w, "decoder.conv1.weight");
    REQ(d->conv1_b, "decoder.conv1.bias");
    for (int b = 0; b < N_BLOCK; b++) {
        dac_block *bl = &d->blocks[b];
        bl->stride = BLOCK_STRIDE[b];
        bl->kernel = BLOCK_KERNEL[b];
        bl->pad = BLOCK_PAD[b];
        bl->out_pad = BLOCK_OUTPAD[b];
        REQ(bl->snake_alpha, "decoder.block.%d.snake1.alpha", b);
        REQ(bl->convt_w, "decoder.block.%d.conv_t1.weight", b);
        REQ(bl->convt_b, "decoder.block.%d.conv_t1.bias", b);
        for (int r = 0; r < 3; r++) {
            REQ(bl->ru_alpha1[r], "decoder.block.%d.res_unit%d.snake1.alpha",
                b, r + 1);
            REQ(bl->ru_conv1_w[r], "decoder.block.%d.res_unit%d.conv1.weight",
                b, r + 1);
            REQ(bl->ru_conv1_b[r], "decoder.block.%d.res_unit%d.conv1.bias",
                b, r + 1);
            REQ(bl->ru_alpha2[r], "decoder.block.%d.res_unit%d.snake2.alpha",
                b, r + 1);
            REQ(bl->ru_conv2_w[r], "decoder.block.%d.res_unit%d.conv2.weight",
                b, r + 1);
            REQ(bl->ru_conv2_b[r], "decoder.block.%d.res_unit%d.conv2.bias",
                b, r + 1);
        }
    }
    REQ(d->final_alpha, "decoder.snake1.alpha");
    REQ(d->final_conv_w, "decoder.conv2.weight");
    REQ(d->final_conv_b, "decoder.conv2.bias");
#undef REQ

    if (!ok) {
        ov_decoder_free(d);
        return NULL;
    }

    d->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(d->backend));
    return d;
}

void ov_decoder_free(ov_decoder *d) {
    if (!d) return;
    if (d->galloc) ggml_gallocr_free(d->galloc);
    if (d->buf) ggml_backend_buffer_free(d->buf);
    if (d->backend) ggml_backend_free(d->backend);
    if (d->wctx) ggml_free(d->wctx);
    free(d);
}

void ov_decoder_set_threads(ov_decoder *d, int n_threads) {
    if (d && d->backend && ggml_backend_is_cpu(d->backend)) {
        ggml_backend_cpu_set_n_threads(d->backend, n_threads);
    }
}

int32_t ov_decoder_sampling_rate(const ov_decoder *d) {
    return d->sampling_rate;
}
int32_t ov_decoder_hop_length(const ov_decoder *d) { return d->hop_length; }
size_t ov_decoder_weights_bytes(const ov_decoder *d) {
    return d->buf ? ggml_backend_buffer_get_size(d->buf) : 0;
}

/* snake(x) = x + (1/(alpha+1e-9)) * sin(alpha*x)^2 ; x: [L, C] */
static struct ggml_tensor *snake(struct ggml_context *ctx,
                                 struct ggml_tensor *x,
                                 struct ggml_tensor *alpha_1c1) {
    int64_t C = alpha_1c1->ne[1];
    struct ggml_tensor *alpha =
        ggml_reshape_2d(ctx, alpha_1c1, 1, C); /* [1, C] */
    struct ggml_tensor *ax = ggml_mul(ctx, x, alpha);
    struct ggml_tensor *s = ggml_sin(ctx, ax);
    struct ggml_tensor *s2 = ggml_mul(ctx, s, s);
    struct ggml_tensor *alpha_eps = ggml_scale_bias(ctx, alpha, 1.0f, 1e-9f);
    return ggml_add(ctx, x, ggml_div(ctx, s2, alpha_eps));
}

/* conv1d + bias; x: [L, Cin] -> [L', Cout] */
static struct ggml_tensor *conv1d_b(struct ggml_context *ctx,
                                    struct ggml_tensor *w,
                                    struct ggml_tensor *b,
                                    struct ggml_tensor *x, int s, int p,
                                    int dil) {
    struct ggml_tensor *x3 = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1);
    struct ggml_tensor *y = ggml_conv_1d(ctx, w, x3, s, p, dil);
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    struct ggml_tensor *bias = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
    return ggml_add(ctx, y, bias);
}

static int64_t ov_decoder_decode_full(ov_decoder *d, const int32_t *codes,
                                      int32_t n_frames, float *out_audio) {
    const int64_t T = n_frames;
    const int64_t n_samples = T * d->hop_length;

    const size_t n_nodes_max = 4096;
    size_t ctx_size = ggml_tensor_overhead() * n_nodes_max +
                      ggml_graph_overhead_custom(n_nodes_max, false);
    struct ggml_init_params ip = {ctx_size, NULL, /*no_alloc=*/true};
    struct ggml_context *ctx = ggml_init(ip);

    struct ggml_tensor *code_in[N_CODEBOOK];
    for (int cb = 0; cb < N_CODEBOOK; cb++) {
        code_in[cb] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(code_in[cb]);
    }

    /* RVQ: q = sum_cb project_out(embed[codes]) ; [1024, T] */
    struct ggml_tensor *q = NULL;
    for (int cb = 0; cb < N_CODEBOOK; cb++) {
        struct ggml_tensor *e =
            ggml_get_rows(ctx, d->rvq[cb].embed, code_in[cb]); /* [64, T] */
        struct ggml_tensor *o = ggml_mul_mat(ctx, d->rvq[cb].proj_w, e);
        struct ggml_tensor *bias =
            ggml_reshape_2d(ctx, d->rvq[cb].proj_b, d->rvq[cb].proj_b->ne[0], 1);
        o = ggml_add(ctx, o, bias);
        q = q ? ggml_add(ctx, q, o) : o;
    }

    /* fc2: [256, T] */
    struct ggml_tensor *x = ggml_mul_mat(ctx, d->fc2_w, q);
    x = ggml_add(ctx, x,
                 ggml_reshape_2d(ctx, d->fc2_b, d->fc2_b->ne[0], 1));

    /* layout de conv: [L, C] */
    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    /* conv1 k7 p3 */
    x = conv1d_b(ctx, d->conv1_w, d->conv1_b, x, 1, 3, 1);

    for (int b = 0; b < N_BLOCK; b++) {
        dac_block *bl = &d->blocks[b];
        x = snake(ctx, x, bl->snake_alpha);

        /* conv transposta: ggml exige p=0; calcula cheio e fatia.
         * O kernel é armazenado em F16 (exigência do backend CPU); o backend
         * CUDA exige F32 para esta op, então convertemos no grafo (5 tensores
         * pequenos, custo desprezível). */
        {
            struct ggml_tensor *convt_w = bl->convt_w;
            if (d->is_gpu) {
                convt_w = ggml_cast(ctx, convt_w, GGML_TYPE_F32);
            }
            struct ggml_tensor *y =
                ggml_conv_transpose_1d(ctx, convt_w, x, bl->stride, 0, 1);
            int64_t full = (x->ne[0] - 1) * bl->stride + bl->kernel;
            int64_t out_len = (x->ne[0] - 1) * bl->stride - 2 * bl->pad +
                              bl->kernel + bl->out_pad;
            (void)full;
            y = ggml_view_2d(ctx, y, out_len, y->ne[1], y->nb[1],
                             (size_t)bl->pad * y->nb[0]);
            y = ggml_cont(ctx, y);
            struct ggml_tensor *bias =
                ggml_reshape_2d(ctx, bl->convt_b, 1, bl->convt_b->ne[0]);
            x = ggml_add(ctx, y, bias);
        }

        for (int r = 0; r < 3; r++) {
            struct ggml_tensor *y = snake(ctx, x, bl->ru_alpha1[r]);
            y = conv1d_b(ctx, bl->ru_conv1_w[r], bl->ru_conv1_b[r], y, 1,
                         RU_PAD[r], RU_DILATION[r]);
            y = snake(ctx, y, bl->ru_alpha2[r]);
            y = conv1d_b(ctx, bl->ru_conv2_w[r], bl->ru_conv2_b[r], y, 1, 0,
                         1);
            x = ggml_add(ctx, x, y);
        }
    }

    x = snake(ctx, x, d->final_alpha);
    x = conv1d_b(ctx, d->final_conv_w, d->final_conv_b, x, 1, 3, 1);
    /* tanh final é Identity() nesta variante */

    ggml_set_output(x);

    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, n_nodes_max, false);
    ggml_build_forward_expand(gf, x);

    if (!ggml_gallocr_alloc_graph(d->galloc, gf)) {
        fprintf(stderr, "ov_decoder: falha ao alocar grafo\n");
        ggml_free(ctx);
        return -1;
    }

    for (int cb = 0; cb < N_CODEBOOK; cb++) {
        ggml_backend_tensor_set(code_in[cb], codes + (size_t)cb * T, 0,
                                (size_t)T * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(d->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ov_decoder: falha no compute\n");
        ggml_free(ctx);
        return -1;
    }

    if (x->ne[0] != n_samples || x->ne[1] != 1) {
        fprintf(stderr,
                "ov_decoder: tamanho inesperado: %lld x %lld (esperado %lld)\n",
                (long long)x->ne[0], (long long)x->ne[1],
                (long long)n_samples);
    }
    int64_t n_out = x->ne[0] < n_samples ? x->ne[0] : n_samples;
    ggml_backend_tensor_get(x, out_audio, 0, (size_t)n_out * sizeof(float));

    ggml_free(ctx);
    return n_out;
}

int64_t ov_decoder_decode(ov_decoder *d, const int32_t *codes,
                          int32_t n_frames, float *out_audio) {
    if (!d || !codes || !out_audio || n_frames <= 0) return -1;

    /* Para CPU ou audio curto, o caminho original continua identico. */
    if (!d->is_gpu || n_frames <= OV_DECODE_CHUNK_FRAMES) {
        return ov_decoder_decode_full(d, codes, n_frames, out_audio);
    }

    const int32_t max_read_frames =
        OV_DECODE_CHUNK_FRAMES + 2 * OV_DECODE_CONTEXT_FRAMES;
    int32_t *chunk_codes = malloc((size_t)N_CODEBOOK * max_read_frames *
                                  sizeof(*chunk_codes));
    float *chunk_audio = malloc((size_t)max_read_frames * d->hop_length *
                                sizeof(*chunk_audio));
    if (!chunk_codes || !chunk_audio) {
        fprintf(stderr, "ov_decoder: memoria insuficiente para decode em blocos\n");
        free(chunk_audio);
        free(chunk_codes);
        return -1;
    }

    int64_t total_out = 0;
    for (int32_t core_start = 0; core_start < n_frames;
         core_start += OV_DECODE_CHUNK_FRAMES) {
        int32_t core_end = core_start + OV_DECODE_CHUNK_FRAMES;
        if (core_end > n_frames) core_end = n_frames;

        int32_t read_start = core_start - OV_DECODE_CONTEXT_FRAMES;
        if (read_start < 0) read_start = 0;
        int32_t read_end = core_end + OV_DECODE_CONTEXT_FRAMES;
        if (read_end > n_frames) read_end = n_frames;
        const int32_t read_frames = read_end - read_start;

        for (int cb = 0; cb < N_CODEBOOK; cb++) {
            memcpy(chunk_codes + (size_t)cb * read_frames,
                   codes + (size_t)cb * n_frames + read_start,
                   (size_t)read_frames * sizeof(*chunk_codes));
        }

        const int64_t decoded = ov_decoder_decode_full(
            d, chunk_codes, read_frames, chunk_audio);
        const int64_t trim_samples =
            (int64_t)(core_start - read_start) * d->hop_length;
        const int64_t copy_samples =
            (int64_t)(core_end - core_start) * d->hop_length;
        if (decoded < trim_samples + copy_samples) {
            fprintf(stderr, "ov_decoder: bloco de audio truncado\n");
            free(chunk_audio);
            free(chunk_codes);
            return -1;
        }

        memcpy(out_audio + total_out, chunk_audio + trim_samples,
               (size_t)copy_samples * sizeof(*out_audio));
        total_out += copy_samples;
    }

    free(chunk_audio);
    free(chunk_codes);
    return total_out;
}
