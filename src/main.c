/* omnivoice-pt — TTS em português com consumo mínimo de VRAM.
 *
 * Pipeline: tokenizer BPE -> prompt -> difusão mascarada (Qwen3/GGML)
 *           -> decoder DAC -> WAV 24 kHz.
 *
 * Uso:
 *   omnivoice-pt -m modelo.gguf -d decoder.gguf -k tokenizer.bin \
 *       -t "Texto em português." -o saida.wav [-p voz.ovprompt] \
 *       [--gpu] [--steps 32] [--seed 42] [--speed 1.0]
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ov_decoder.h"
#include "ov_model.h"
#include "ov_platform.h"
#include "ov_sampler.h"
#include "ov_tokenizer.h"

/* ------------------------------------------------------------------ */
/* Estimador de duração (port de RuleDurationEstimator para PT/latim)  */
/* ------------------------------------------------------------------ */

static uint32_t utf8_next_cp(const char *s, size_t *i) {
    uint8_t c = (uint8_t)s[*i];
    if (c < 0x80) {
        (*i)++;
        return c;
    }
    int extra = (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : 3;
    uint32_t cp = c & (0x3F >> extra);
    for (int k = 1; k <= extra && s[*i + k]; k++) {
        cp = (cp << 6) | ((uint8_t)s[*i + k] & 0x3F);
    }
    *i += 1 + extra;
    return cp;
}

static float char_weight(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return 1.0f;
    if (cp == ' ') return 0.2f;
    if (cp >= '0' && cp <= '9') return 3.5f;
    /* pontuação e símbolos ASCII */
    if (cp < 0x80) return 0.5f;
    /* letras latinas acentuadas (PT) */
    if ((cp >= 0xC0 && cp <= 0xFF) && cp != 0xD7 && cp != 0xF7) return 1.0f;
    if (cp >= 0x100 && cp <= 0x2AF) return 1.0f; /* latim estendido */
    /* marcas combinantes */
    if (cp >= 0x300 && cp <= 0x36F) return 0.0f;
    /* CJK etc. — improvável em PT */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 3.0f;
    /* pontuação tipográfica geral (–—“”…) */
    if (cp >= 0x2000 && cp <= 0x206F) return 0.5f;
    return 1.0f;
}

static float text_weight(const char *s) {
    float w = 0.0f;
    size_t i = 0;
    while (s[i]) w += char_weight(utf8_next_cp(s, &i));
    return w;
}

/* retorna nº de frames de áudio (25 Hz) estimado */
static int32_t estimate_target_frames(const char *text, const char *ref_text,
                                      int32_t ref_frames, float speed) {
    if (!ref_text || !ref_text[0] || ref_frames <= 0) {
        ref_text = "Nice to meet you.";
        ref_frames = 25;
    }
    float ref_w = text_weight(ref_text);
    if (ref_w <= 0) return 1;
    float est = text_weight(text) * (float)ref_frames / ref_w;
    /* boost para textos curtos (low_threshold=50, boost=3) */
    if (est < 50.0f) est = 50.0f * powf(est / 50.0f, 1.0f / 3.0f);
    if (speed > 0 && speed != 1.0f) est /= speed;
    int32_t n = (int32_t)est;
    return n < 1 ? 1 : n;
}

/* ------------------------------------------------------------------ */
/* Voice prompt (.ovprompt)                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    int32_t n_codebook, n_frames;
    float ref_rms;
    char *ref_text;
    int32_t *tokens; /* [C * T] */
} ov_prompt;

static ov_prompt *prompt_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "nao abriu prompt %s\n", path);
        return NULL;
    }
    char magic[4];
    uint32_t ver, C, T, text_len;
    float rms;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "OVPR", 4) != 0 ||
        fread(&ver, 4, 1, f) != 1 || fread(&C, 4, 1, f) != 1 ||
        fread(&T, 4, 1, f) != 1 || fread(&rms, 4, 1, f) != 1 ||
        fread(&text_len, 4, 1, f) != 1) {
        fprintf(stderr, "prompt invalido\n");
        fclose(f);
        return NULL;
    }
    ov_prompt *p = calloc(1, sizeof(*p));
    p->n_codebook = (int32_t)C;
    p->n_frames = (int32_t)T;
    p->ref_rms = rms;
    p->ref_text = malloc((size_t)text_len + 1);
    if (fread(p->ref_text, 1, text_len, f) != text_len) goto fail;
    p->ref_text[text_len] = 0;
    {
        uint16_t *tmp = malloc((size_t)C * T * 2);
        if (fread(tmp, 2, (size_t)C * T, f) != (size_t)C * T) {
            free(tmp);
            goto fail;
        }
        p->tokens = malloc((size_t)C * T * 4);
        for (size_t i = 0; i < (size_t)C * T; i++) p->tokens[i] = tmp[i];
        free(tmp);
    }
    fclose(f);
    return p;
fail:
    fprintf(stderr, "prompt truncado\n");
    fclose(f);
    free(p->ref_text);
    free(p);
    return NULL;
}

static void prompt_free(ov_prompt *p) {
    if (!p) return;
    free(p->ref_text);
    free(p->tokens);
    free(p);
}

/* ------------------------------------------------------------------ */
/* WAV                                                                 */
/* ------------------------------------------------------------------ */

static int wav_write(const char *path, const float *pcm, int64_t n,
                     int32_t rate) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t data_bytes = (uint32_t)(n * 2);
    uint32_t riff = 36 + data_bytes;
    uint16_t fmt16 = 1, ch = 1, bits = 16;
    uint32_t byte_rate = (uint32_t)rate * 2;
    uint16_t block = 2;
    fwrite("RIFF", 1, 4, f);
    fwrite(&riff, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&fmt16, 2, 1, f);
    fwrite(&ch, 2, 1, f);
    fwrite(&rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_bytes, 4, 1, f);
    for (int64_t i = 0; i < n; i++) {
        float v = pcm[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int16_t s = (int16_t)lrintf(v * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */

static void usage(const char *prog) {
    fprintf(stderr,
            "uso: %s -m modelo.gguf -d decoder.gguf -k tokenizer.bin\n"
            "        -t \"texto\" -o saida.wav [-p voz.ovprompt]\n"
            "        [--gpu] [--threads N] [--steps N] [--seed N] "
            "[--speed F]\n"
            "        [--t_shift F] [--cfg F] [--temp F]\n"
            "\n"
            "  --steps N    passos de difusao (padrao 32)\n"
            "  --t_shift F  concentra os passos no fim; menor = mais tardio\n"
            "               (padrao 0.10). Com 0.10 e 32 passos, 22%% dos\n"
            "               tokens sao decididos no ultimo passo; com 12\n"
            "               passos sao 47%% — subir t_shift distribui melhor\n"
            "               e permite cortar passos. Ex.: --steps 16"
            " --t_shift 0.3\n"
            "  --cfg F      guidance_scale (padrao 2.0; 0 desliga o forward\n"
            "               uncond e fica ~30%% mais rapido, com menos\n"
            "               aderencia ao texto)\n"
            "  --temp F     class_temperature (padrao 0 = greedy). >0 sorteia\n"
            "               entre os 10%% melhores tokens: mais variacao de\n"
            "               prosodia, mesmo custo\n",
            prog);
}

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <io.h>
#include <fcntl.h>
/* argv do Windows chega em ANSI (CP1252); converte o command line
 * wide para UTF-8 para que "é" não vire byte solto 0xE9 */
static char **win_utf8_argv(int *out_argc) {
    int argc;
    wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    char **argv = malloc((size_t)(argc + 1) * sizeof(char *));
    for (int i = 0; i < argc; i++) {
        int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL,
                                    NULL);
        argv[i] = malloc((size_t)n);
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, argv[i], n, NULL, NULL);
    }
    argv[argc] = NULL;
    LocalFree(wargv);
    *out_argc = argc;
    return argv;
}
#endif

/* Resultado de uma geração. */
typedef struct {
    int ok;
    double gen_s, dec_s, audio_s, rtf;
    char err[256];
} gen_result;

/* Gera um áudio (tk/m/dec já carregados). Usada tanto pelo modo CLI quanto
 * pelo modo servidor (chamada repetidamente com o modelo residente). */
static gen_result generate_one(ov_tokenizer *tk, ov_model *m, ov_decoder *dec,
                               const char *text, const char *prompt_path,
                               const char *out_path, ov_sample_config scfg,
                               float speed) {
    gen_result r = {0};
    const ov_model_params *p = ov_model_get_params(m);

    ov_prompt *vp = NULL;
    if (prompt_path && strcmp(prompt_path, "-") != 0) {
        vp = prompt_load(prompt_path);
        if (!vp) {
            snprintf(r.err, sizeof(r.err), "falha ao carregar voz %s",
                     prompt_path);
            return r;
        }
    }

    char style[256];
    snprintf(style, sizeof(style),
             "%s<|lang_start|>pt<|lang_end|>"
             "<|instruct_start|>None<|instruct_end|>",
             vp ? "<|denoise|>" : "");

    char full_text[8192];
    if (vp) {
        snprintf(full_text, sizeof(full_text),
                 "<|text_start|>%s %s<|text_end|>", vp->ref_text, text);
    } else {
        snprintf(full_text, sizeof(full_text), "<|text_start|>%s<|text_end|>",
                 text);
    }

    int32_t style_ids[128], text_ids[4096];
    int32_t n_style = ov_tokenizer_encode(tk, style, style_ids, 128);
    int32_t n_text = ov_tokenizer_encode(tk, full_text, text_ids, 4096);
    if (n_style < 0 || n_text < 0) {
        snprintf(r.err, sizeof(r.err), "texto longo demais");
        if (vp) prompt_free(vp);
        return r;
    }

    int32_t target = estimate_target_frames(
        text, vp ? vp->ref_text : NULL, vp ? vp->n_frames : 0, speed);
    if (target > 750) target = 750;

    const int32_t C = p->n_codebook;
    const int32_t n_txt = n_style + n_text;
    const int32_t ref_len = vp ? vp->n_frames : 0;
    const int32_t prefix_len = n_txt + ref_len;

    int32_t *prefix = malloc((size_t)C * prefix_len * sizeof(int32_t));
    uint8_t *pmask = calloc(1, (size_t)prefix_len);

    for (int32_t i = 0; i < n_txt; i++) {
        int32_t orig = i < n_style ? style_ids[i] : text_ids[i - n_style];
        int32_t re = ov_model_remap_token(m, orig);
        if (re < 0) {
            snprintf(r.err, sizeof(r.err), "token %d fora do keep-set PT",
                     orig);
            free(prefix);
            free(pmask);
            if (vp) prompt_free(vp);
            return r;
        }
        for (int32_t cb = 0; cb < C; cb++) {
            prefix[(size_t)cb * prefix_len + i] = re;
        }
    }
    for (int32_t t = 0; t < ref_len; t++) {
        pmask[n_txt + t] = 1;
        for (int32_t cb = 0; cb < C; cb++) {
            prefix[(size_t)cb * prefix_len + n_txt + t] =
                vp->tokens[(size_t)cb * ref_len + t];
        }
    }

    double t1 = ov_time_seconds();
    int32_t *tokens = malloc((size_t)C * target * sizeof(int32_t));
    if (ov_sample(m, prefix, pmask, prefix_len, target, &scfg, tokens) != 0) {
        snprintf(r.err, sizeof(r.err), "falha na geracao");
        free(prefix);
        free(pmask);
        free(tokens);
        if (vp) prompt_free(vp);
        return r;
    }
    r.gen_s = ov_time_seconds() - t1;

    double t2 = ov_time_seconds();
    int64_t n_samples = (int64_t)target * ov_decoder_hop_length(dec);
    float *pcm = malloc((size_t)n_samples * sizeof(float));
    int64_t n = ov_decoder_decode(dec, tokens, target, pcm);
    r.dec_s = ov_time_seconds() - t2;
    if (n < 0) {
        snprintf(r.err, sizeof(r.err), "falha no decode");
        free(prefix);
        free(pmask);
        free(tokens);
        free(pcm);
        if (vp) prompt_free(vp);
        return r;
    }

    if (vp && vp->ref_rms < 0.1f) {
        for (int64_t i = 0; i < n; i++) pcm[i] *= vp->ref_rms / 0.1f;
    } else if (!vp) {
        float peak = 0;
        for (int64_t i = 0; i < n; i++) {
            float a = fabsf(pcm[i]);
            if (a > peak) peak = a;
        }
        if (peak > 1e-6f) {
            float s = 0.5f / peak;
            for (int64_t i = 0; i < n; i++) pcm[i] *= s;
        }
    }

    if (wav_write(out_path, pcm, n, ov_decoder_sampling_rate(dec)) != 0) {
        snprintf(r.err, sizeof(r.err), "falha ao escrever %s", out_path);
        free(prefix);
        free(pmask);
        free(tokens);
        free(pcm);
        if (vp) prompt_free(vp);
        return r;
    }

    r.audio_s = (double)n / ov_decoder_sampling_rate(dec);
    r.rtf = (r.gen_s + r.dec_s) / r.audio_s;
    r.ok = 1;

    free(prefix);
    free(pmask);
    free(tokens);
    free(pcm);
    if (vp) prompt_free(vp);
    return r;
}

/* Lê uma linha do stdin (UTF-8) sem tradução; retorna 0 no EOF. */
static int read_line(char *buf, int cap) {
    if (!fgets(buf, cap, stdin)) return 0;
    size_t len = strlen(buf);
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) {
        buf[--len] = 0;
    }
    return 1;
}

/* Modo servidor (worker persistente): mantém o modelo carregado e processa
 * um job por linha do stdin. Cada job:
 *   <out_path>\t<steps>\t<seed>\t<prompt_path|->\t<texto>\n
 * Resposta por linha no stdout:
 *   OK\t<gen_s>\t<dec_s>\t<audio_s>\t<rtf>\t<vram_mb>\n   ou   ERR\t<msg>\n
 * O texto não pode conter TAB/quebra de linha (o cliente troca por espaço).
 * `base`/`speed` vêm da linha de comando do worker e valem para todos os jobs;
 * só steps e seed são por job (o protocolo não mudou). */
static int serve_loop(ov_tokenizer *tk, ov_model *m, ov_decoder *dec,
                      ov_sample_config base, float speed) {
    int vram_mb = (int)((ov_model_weights_bytes(m) +
                         ov_decoder_weights_bytes(dec)) /
                            1000000 +
                        60 /* estimativa de ativações */);
    fprintf(stdout, "READY\t%d\n", vram_mb);
    fflush(stdout);
    fprintf(stderr, "worker pronto (VRAM ~%d MB), aguardando jobs...\n",
            vram_mb);

    char *line = malloc(16384);
    while (read_line(line, 16384)) {
        if (line[0] == 0) continue;

        /* separa os 4 primeiros campos por TAB; o texto é o resto */
        char *fields[4];
        char *cursor = line;
        int nf = 0;
        for (; nf < 4; nf++) {
            char *tab = strchr(cursor, '\t');
            if (!tab) break;
            *tab = 0;
            fields[nf] = cursor;
            cursor = tab + 1;
        }
        if (nf < 4) {
            fprintf(stdout, "ERR\tjob mal formado\n");
            fflush(stdout);
            continue;
        }
        const char *out_path = fields[0];
        int steps = atoi(fields[1]);
        uint64_t seed = (uint64_t)atoll(fields[2]);
        const char *prompt_path = fields[3];
        const char *text = cursor;

        ov_sample_config scfg = base;
        if (steps > 0) scfg.num_step = steps;
        scfg.seed = seed;

        gen_result r = generate_one(tk, m, dec, text, prompt_path, out_path,
                                    scfg, speed);
        if (r.ok) {
            fprintf(stdout, "OK\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n", r.gen_s,
                    r.dec_s, r.audio_s, r.rtf, vram_mb);
        } else {
            fprintf(stdout, "ERR\t%s\n", r.err);
        }
        fflush(stdout);
    }
    free(line);
    return 0;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    argv = win_utf8_argv(&argc);
#endif
    const char *model_path = NULL, *dec_path = NULL, *tok_path = NULL;
    const char *text = NULL, *out_path = NULL, *prompt_path = NULL;
    int use_gpu = 0, serve = 0, n_threads = 0;
    ov_sample_config scfg = ov_sample_config_default();
    float speed = 1.0f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) dec_path = argv[++i];
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) text = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt_path = argv[++i];
        else if (!strcmp(argv[i], "--gpu")) use_gpu = 1;
        else if (!strcmp(argv[i], "--serve")) serve = 1;
        else if ((!strcmp(argv[i], "--threads") || !strcmp(argv[i], "-n")) && i + 1 < argc)
            n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc)
            scfg.num_step = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            scfg.seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--speed") && i + 1 < argc)
            speed = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--t_shift") && i + 1 < argc)
            scfg.t_shift = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--cfg") && i + 1 < argc)
            scfg.guidance_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc)
            scfg.class_temperature = (float)atof(argv[++i]);
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!model_path || !dec_path || !tok_path) {
        usage(argv[0]);
        return 2;
    }
    if (!serve && (!text || !out_path)) {
        usage(argv[0]);
        return 2;
    }
    /* t_shift = 0 zera o denominador do cronograma em t=1 */
    if (!(scfg.t_shift > 0.0f) || scfg.num_step < 1 ||
        scfg.guidance_scale < 0.0f || scfg.class_temperature < 0.0f) {
        fprintf(stderr,
                "erro: --t_shift deve ser > 0, --steps >= 1, --cfg e --temp "
                ">= 0\n");
        return 2;
    }

    /* Em CPU, sem --threads explícito, escolhe um bom padrão: ~metade dos
     * processadores lógicos (≈ núcleos físicos), limitado a [2, 16] — a
     * geração satura em ~16 threads, e usar todos os lógicos com
     * hyperthreading rende pouco. A outra IA deixava fixo em 4. */
    if (!use_gpu && n_threads <= 0) {
        int logical = ov_logical_cpu_count();
        n_threads = logical / 2;
        if (n_threads < 2) n_threads = 2;
        if (n_threads > 16) n_threads = 16;
    }

#ifdef _OPENMP
    /* --threads também governa o laço paralelo do sampler; sem ele, o padrão
     * do OpenMP (todos os processadores lógicos) é o que queremos na GPU, onde
     * o sampler é o único trabalho de CPU. */
    if (n_threads > 0) omp_set_num_threads(n_threads);
#endif

    double t0 = ov_time_seconds();
    ov_tokenizer *tk = ov_tokenizer_load(tok_path);
    if (!tk) return 1;
    ov_model *m = ov_model_load(model_path, use_gpu);
    if (!m) return 1;
    ov_decoder *dec = ov_decoder_load(dec_path, use_gpu);
    if (!dec) return 1;
    if (n_threads > 0) {
        ov_model_set_threads(m, n_threads);
        ov_decoder_set_threads(dec, n_threads);
    }
    if (use_gpu) {
        fprintf(stderr, "[%.1fs] modelos carregados (GPU)\n",
                ov_time_seconds() - t0);
    } else {
        fprintf(stderr, "[%.1fs] modelos carregados (CPU, %d threads)\n",
                ov_time_seconds() - t0, n_threads);
    }

    fprintf(stderr,
            "sampler: steps=%d t_shift=%.2f cfg=%.2f temp=%.2f seed=%llu"
#ifdef _OPENMP
            " (omp %d threads)"
#endif
            "\n",
            scfg.num_step, scfg.t_shift, scfg.guidance_scale,
            scfg.class_temperature, (unsigned long long)scfg.seed
#ifdef _OPENMP
            ,
            omp_get_max_threads()
#endif
    );

    int rc = 0;
    if (serve) {
#ifdef _WIN32
        /* modo binário: evita tradução \n<->\r\n que corromperia o
         * protocolo e os bytes UTF-8 do texto */
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        rc = serve_loop(tk, m, dec, scfg, speed);
    } else {
        gen_result r =
            generate_one(tk, m, dec, text, prompt_path, out_path, scfg, speed);
        if (r.ok) {
            fprintf(stderr,
                    "ok: %s (%.1fs de audio) | geracao %.2fs, decode %.2fs, "
                    "RTF %.2f\n",
                    out_path, r.audio_s, r.gen_s, r.dec_s, r.rtf);
        } else {
            fprintf(stderr, "erro: %s\n", r.err);
            rc = 1;
        }
    }

    ov_tokenizer_free(tk);
    ov_model_free(m);
    ov_decoder_free(dec);
    return rc;
}
