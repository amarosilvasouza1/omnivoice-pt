/* Implementação do tokenizer BPE byte-level do Qwen3 (ver ov_tokenizer.h). */

#include "ov_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Estruturas                                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t key; /* (left << 32) | right; UINT64_MAX = vazio */
    int32_t rank;
    int32_t result;
} pair_slot;

typedef struct {
    uint32_t hash;
    int32_t id; /* -1 = vazio */
} str_slot;

struct ov_tokenizer {
    int32_t n_vocab;
    char **strs;        /* id -> string UTF-8 (forma unicode do BPE) */
    uint16_t *str_lens; /* id -> comprimento em bytes */

    str_slot *str_map; /* hash string -> id */
    uint32_t str_map_size;

    pair_slot *pair_map; /* hash (l,r) -> merge */
    uint32_t pair_map_size;

    int32_t byte_id[256]; /* byte -> id do token de 1 byte */

    int32_t *special_ids; /* added_tokens: casados atomicamente no texto */
    uint32_t n_special;
};

/* ------------------------------------------------------------------ */
/* Hashes                                                              */
/* ------------------------------------------------------------------ */

static uint32_t fnv1a(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h ? h : 1u;
}

static uint32_t next_pow2(uint32_t x) {
    uint32_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

static void str_map_insert(ov_tokenizer *tk, const char *s, uint16_t len,
                           int32_t id) {
    uint32_t h = fnv1a(s, len);
    uint32_t mask = tk->str_map_size - 1;
    uint32_t i = h & mask;
    while (tk->str_map[i].id >= 0) {
        int32_t other = tk->str_map[i].id;
        if (tk->str_map[i].hash == h && tk->str_lens[other] == len &&
            memcmp(tk->strs[other], s, len) == 0) {
            return; /* duplicado: mantém o primeiro id */
        }
        i = (i + 1) & mask;
    }
    tk->str_map[i].hash = h;
    tk->str_map[i].id = id;
}

static int32_t str_map_find(const ov_tokenizer *tk, const char *s, size_t len) {
    uint32_t h = fnv1a(s, len);
    uint32_t mask = tk->str_map_size - 1;
    uint32_t i = h & mask;
    while (tk->str_map[i].id >= 0) {
        int32_t id = tk->str_map[i].id;
        if (tk->str_map[i].hash == h && tk->str_lens[id] == len &&
            memcmp(tk->strs[id], s, len) == 0) {
            return id;
        }
        i = (i + 1) & mask;
    }
    return -1;
}

static void pair_map_insert(ov_tokenizer *tk, uint64_t key, int32_t rank,
                            int32_t result) {
    uint32_t mask = tk->pair_map_size - 1;
    uint32_t i = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 32) & mask;
    while (tk->pair_map[i].key != UINT64_MAX) {
        if (tk->pair_map[i].key == key) return; /* primeiro rank vence */
        i = (i + 1) & mask;
    }
    tk->pair_map[i].key = key;
    tk->pair_map[i].rank = rank;
    tk->pair_map[i].result = result;
}

static const pair_slot *pair_map_find(const ov_tokenizer *tk, int32_t l,
                                      int32_t r) {
    uint64_t key = ((uint64_t)(uint32_t)l << 32) | (uint32_t)r;
    uint32_t mask = tk->pair_map_size - 1;
    uint32_t i = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 32) & mask;
    while (tk->pair_map[i].key != UINT64_MAX) {
        if (tk->pair_map[i].key == key) return &tk->pair_map[i];
        i = (i + 1) & mask;
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* bytes_to_unicode (GPT-2): byte -> codepoint                         */
/* ------------------------------------------------------------------ */

static void build_byte_to_cp(uint32_t out[256]) {
    int used[256] = {0};
    for (int b = 0x21; b <= 0x7E; b++) used[b] = 1;
    for (int b = 0xA1; b <= 0xAC; b++) used[b] = 1;
    for (int b = 0xAE; b <= 0xFF; b++) used[b] = 1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        out[b] = used[b] ? (uint32_t)b : (uint32_t)(256 + n++);
    }
}

static size_t cp_to_utf8(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Decodifica um codepoint UTF-8; avança *i. Bytes inválidos viram 0xFFFD. */
static uint32_t utf8_next(const char *s, size_t len, size_t *i) {
    uint8_t c = (uint8_t)s[*i];
    if (c < 0x80) {
        (*i)++;
        return c;
    }
    uint32_t cp;
    int extra;
    if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F;
        extra = 1;
    } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F;
        extra = 2;
    } else if ((c & 0xF8) == 0xF0) {
        cp = c & 0x07;
        extra = 3;
    } else {
        (*i)++;
        return 0xFFFD;
    }
    if (*i + 1 + extra > len) {
        (*i)++;
        return 0xFFFD;
    }
    for (int k = 1; k <= extra; k++) {
        uint8_t cc = (uint8_t)s[*i + k];
        if ((cc & 0xC0) != 0x80) {
            (*i)++;
            return 0xFFFD;
        }
        cp = (cp << 6) | (cc & 0x3F);
    }
    *i += 1 + extra;
    return cp;
}

/* ------------------------------------------------------------------ */
/* Classes Unicode (subconjunto suficiente: latim completo + grandes   */
/* alfabetos; validado contra os goldens do tokenizer HF)              */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t lo, hi;
} cp_range;

static const cp_range LETTER_RANGES[] = {
    {0x0041, 0x005A}, {0x0061, 0x007A}, {0x00AA, 0x00AA}, {0x00B5, 0x00B5},
    {0x00BA, 0x00BA}, {0x00C0, 0x00D6}, {0x00D8, 0x00F6}, {0x00F8, 0x02C1},
    {0x02C6, 0x02D1}, {0x02E0, 0x02E4}, {0x0370, 0x0374}, {0x0376, 0x0377},
    {0x037A, 0x037D}, {0x0386, 0x0386}, {0x0388, 0x03F5}, {0x03F7, 0x0481},
    {0x048A, 0x052F}, {0x0531, 0x0556}, {0x0561, 0x0587}, {0x05D0, 0x05EA},
    {0x0620, 0x064A}, {0x0671, 0x06D3}, {0x0904, 0x0939}, {0x0958, 0x0961},
    {0x0E01, 0x0E30}, {0x10A0, 0x10C5}, {0x1E00, 0x1FBC}, {0x1FC2, 0x1FCC},
    {0x1FD0, 0x1FDB}, {0x1FE0, 0x1FEC}, {0x1FF2, 0x1FFC}, {0x2C60, 0x2C7F},
    {0x3041, 0x3096}, {0x30A1, 0x30FA}, {0x3105, 0x312F}, {0x3400, 0x4DBF},
    {0x4E00, 0x9FFF}, {0xA720, 0xA7FF}, {0xAC00, 0xD7A3}, {0xF900, 0xFAD9},
    {0xFB00, 0xFB06},
};

static const cp_range DIGIT_RANGES[] = {
    {0x0030, 0x0039}, {0x0660, 0x0669}, {0x06F0, 0x06F9}, {0x0966, 0x096F},
    {0x09E6, 0x09EF}, {0x0E50, 0x0E59}, {0xFF10, 0xFF19},
};

static int in_ranges(uint32_t cp, const cp_range *r, size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo) {
            hi = mid;
        } else if (cp > r[mid].hi) {
            lo = mid + 1;
        } else {
            return 1;
        }
    }
    return 0;
}

static int is_letter(uint32_t cp) {
    return in_ranges(cp, LETTER_RANGES,
                     sizeof(LETTER_RANGES) / sizeof(LETTER_RANGES[0]));
}

static int is_digit(uint32_t cp) {
    return in_ranges(cp, DIGIT_RANGES,
                     sizeof(DIGIT_RANGES) / sizeof(DIGIT_RANGES[0]));
}

static int is_ws(uint32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D:
        case 0x20: case 0x85: case 0xA0: case 0x1680:
        case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return 1;
        default:
            return (cp >= 0x2000 && cp <= 0x200A);
    }
}

static int is_crlf(uint32_t cp) { return cp == 0x0A || cp == 0x0D; }

/* ------------------------------------------------------------------ */
/* Pre-tokenizador: regex do Qwen2                                     */
/*   (?i:'s|'t|'re|'ve|'m|'ll|'d)                                      */
/*   |[^\r\n\p{L}\p{N}]?\p{L}+                                         */
/*   |\p{N}                                                            */
/*   | ?[^\s\p{L}\p{N}]+[\r\n]*                                        */
/*   |\s*[\r\n]+                                                       */
/*   |\s+(?!\S)                                                        */
/*   |\s+                                                              */
/* Alternativas testadas em ordem; retorna o comprimento em bytes do   */
/* próximo pre-token a partir de `pos`.                                */
/* ------------------------------------------------------------------ */

static size_t pretok_next(const char *s, size_t len, size_t pos) {
    size_t i = pos;
    uint32_t cp = utf8_next(s, len, &i);
    size_t cp_end = i;

    /* (a) contrações inglesas com apóstrofo ASCII */
    if (cp == '\'' && cp_end < len) {
        const char *rest = s + cp_end;
        size_t rem = len - cp_end;
        static const char *ONE = "stmd";
        if (rem >= 1) {
            char c0 = (char)(rest[0] | 0x20);
            if (rem >= 2) {
                char c1 = (char)(rest[1] | 0x20);
                if ((c0 == 'r' && c1 == 'e') || (c0 == 'v' && c1 == 'e') ||
                    (c0 == 'l' && c1 == 'l')) {
                    return cp_end + 2 - pos;
                }
            }
            if (strchr(ONE, c0) != NULL) {
                return cp_end + 1 - pos;
            }
        }
    }

    /* (b) [^\r\n\p{L}\p{N}]?\p{L}+ */
    {
        size_t j = pos;
        size_t k = j;
        uint32_t c0 = utf8_next(s, len, &k);
        size_t after0 = k;
        int has_prefix = 0;
        if (!is_crlf(c0) && !is_letter(c0) && !is_digit(c0)) {
            has_prefix = 1;
            j = after0;
        }
        /* precisa de 1+ letras a partir de j */
        size_t m = j;
        size_t last_ok = j;
        int n_letters = 0;
        while (m < len) {
            size_t t = m;
            uint32_t c = utf8_next(s, len, &t);
            if (!is_letter(c)) break;
            n_letters++;
            m = t;
            last_ok = m;
        }
        if (n_letters > 0) {
            return last_ok - pos;
        }
        (void)has_prefix;
    }

    /* (c) um dígito */
    if (is_digit(cp)) {
        return cp_end - pos;
    }

    /* (d)  ?[^\s\p{L}\p{N}]+[\r\n]* */
    {
        size_t j = pos;
        if (cp == ' ') j = cp_end;
        size_t m = j;
        int n_punct = 0;
        while (m < len) {
            size_t t = m;
            uint32_t c = utf8_next(s, len, &t);
            if (is_ws(c) || is_letter(c) || is_digit(c)) break;
            n_punct++;
            m = t;
        }
        if (n_punct > 0) {
            /* [\r\n]* */
            while (m < len) {
                size_t t = m;
                uint32_t c = utf8_next(s, len, &t);
                if (!is_crlf(c)) break;
                m = t;
            }
            return m - pos;
        }
    }

    /* run de whitespace a partir de pos (usado por e, f, g) */
    if (is_ws(cp)) {
        size_t m = pos;
        size_t last_nl_end = 0;
        size_t run_end = pos;
        while (m < len) {
            size_t t = m;
            uint32_t c = utf8_next(s, len, &t);
            if (!is_ws(c)) break;
            if (is_crlf(c)) last_nl_end = t;
            m = t;
            run_end = m;
        }
        /* (e) \s*[\r\n]+ : casa até o último \r|\n do run */
        if (last_nl_end > 0) {
            return last_nl_end - pos;
        }
        /* (f) \s+(?!\S) : run até o fim da string */
        if (run_end == len) {
            return run_end - pos;
        }
        /* (f) com backtrack: deixa o último ws para o próximo token */
        size_t n_ws_bytes = run_end - pos;
        /* recua um codepoint */
        size_t last_cp_start = pos;
        size_t scan = pos;
        while (scan < run_end) {
            last_cp_start = scan;
            utf8_next(s, len, &scan);
        }
        if (last_cp_start > pos) {
            return last_cp_start - pos;
        }
        /* (g) \s+ : um único ws seguido de não-ws */
        return n_ws_bytes;
    }

    /* fallback: nunca deveria acontecer (toda entrada cai numa classe);
     * consome um codepoint para garantir progresso */
    return cp_end - pos;
}

/* ------------------------------------------------------------------ */
/* BPE                                                                 */
/* ------------------------------------------------------------------ */

#define OV_MAX_PRETOK 512 /* codepoints máximos por pre-token */

static int32_t bpe_encode_pretok(const ov_tokenizer *tk, const char *bytes,
                                 size_t n_bytes, int32_t *out,
                                 int32_t max_out) {
    int32_t ids[OV_MAX_PRETOK];
    int32_t n = 0;
    if (n_bytes > OV_MAX_PRETOK) return -1;
    for (size_t i = 0; i < n_bytes; i++) {
        ids[n++] = tk->byte_id[(uint8_t)bytes[i]];
    }

    /* merge greedy: sempre o par de menor rank */
    while (n > 1) {
        int32_t best_rank = INT32_MAX;
        int32_t best_pos = -1;
        int32_t best_result = -1;
        for (int32_t i = 0; i < n - 1; i++) {
            const pair_slot *p = pair_map_find(tk, ids[i], ids[i + 1]);
            if (p && p->rank < best_rank) {
                best_rank = p->rank;
                best_pos = i;
                best_result = p->result;
            }
        }
        if (best_pos < 0) break;
        ids[best_pos] = best_result;
        memmove(&ids[best_pos + 1], &ids[best_pos + 2],
                (size_t)(n - best_pos - 2) * sizeof(int32_t));
        n--;
    }

    if (n > max_out) return -1;
    memcpy(out, ids, (size_t)n * sizeof(int32_t));
    return n;
}

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */

ov_tokenizer *ov_tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ov_tokenizer: nao abriu %s\n", path);
        return NULL;
    }

    char magic[4];
    uint32_t version, n_vocab, n_merges;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "OVTK", 4) != 0 ||
        fread(&version, 4, 1, f) != 1 || fread(&n_vocab, 4, 1, f) != 1 ||
        fread(&n_merges, 4, 1, f) != 1) {
        fprintf(stderr, "ov_tokenizer: header invalido\n");
        fclose(f);
        return NULL;
    }

    ov_tokenizer *tk = calloc(1, sizeof(*tk));
    tk->n_vocab = (int32_t)n_vocab;
    tk->strs = calloc(n_vocab, sizeof(char *));
    tk->str_lens = calloc(n_vocab, sizeof(uint16_t));

    tk->str_map_size = next_pow2(n_vocab * 2);
    tk->str_map = malloc((size_t)tk->str_map_size * sizeof(str_slot));
    for (uint32_t i = 0; i < tk->str_map_size; i++) tk->str_map[i].id = -1;

    for (uint32_t i = 0; i < n_vocab; i++) {
        uint16_t slen;
        if (fread(&slen, 2, 1, f) != 1) goto fail;
        tk->strs[i] = malloc((size_t)slen + 1);
        if (slen && fread(tk->strs[i], 1, slen, f) != slen) goto fail;
        tk->strs[i][slen] = 0;
        tk->str_lens[i] = slen;
        if (slen) str_map_insert(tk, tk->strs[i], slen, (int32_t)i);
    }

    tk->pair_map_size = next_pow2(n_merges * 2);
    tk->pair_map = malloc((size_t)tk->pair_map_size * sizeof(pair_slot));
    for (uint32_t i = 0; i < tk->pair_map_size; i++) {
        tk->pair_map[i].key = UINT64_MAX;
    }

    for (uint32_t i = 0; i < n_merges; i++) {
        uint32_t l, r, res;
        if (fread(&l, 4, 1, f) != 1 || fread(&r, 4, 1, f) != 1 ||
            fread(&res, 4, 1, f) != 1) {
            goto fail;
        }
        uint64_t key = ((uint64_t)l << 32) | r;
        pair_map_insert(tk, key, (int32_t)i, (int32_t)res);
    }

    if (version >= 2) {
        uint32_t n_special;
        if (fread(&n_special, 4, 1, f) != 1) goto fail;
        tk->n_special = n_special;
        tk->special_ids = malloc((size_t)n_special * sizeof(int32_t));
        for (uint32_t i = 0; i < n_special; i++) {
            uint32_t sid;
            if (fread(&sid, 4, 1, f) != 1) goto fail;
            tk->special_ids[i] = (int32_t)sid;
        }
    }
    fclose(f);

    /* byte -> id */
    uint32_t byte_cp[256];
    build_byte_to_cp(byte_cp);
    for (int b = 0; b < 256; b++) {
        char buf[4];
        size_t blen = cp_to_utf8(byte_cp[b], buf);
        tk->byte_id[b] = str_map_find(tk, buf, blen);
        if (tk->byte_id[b] < 0) {
            fprintf(stderr, "ov_tokenizer: byte-token %d ausente\n", b);
            ov_tokenizer_free(tk);
            return NULL;
        }
    }
    return tk;

fail:
    fprintf(stderr, "ov_tokenizer: arquivo truncado\n");
    fclose(f);
    ov_tokenizer_free(tk);
    return NULL;
}

void ov_tokenizer_free(ov_tokenizer *tk) {
    if (!tk) return;
    if (tk->strs) {
        for (int32_t i = 0; i < tk->n_vocab; i++) free(tk->strs[i]);
        free(tk->strs);
    }
    free(tk->str_lens);
    free(tk->str_map);
    free(tk->pair_map);
    free(tk->special_ids);
    free(tk);
}

/* Tokeniza um trecho SEM tokens especiais. */
static int32_t encode_plain(const ov_tokenizer *tk, const char *text,
                            size_t len, int32_t *out_ids, int32_t max_ids) {
    size_t pos = 0;
    int32_t n_out = 0;
    while (pos < len) {
        size_t tok_len = pretok_next(text, len, pos);
        if (tok_len == 0) tok_len = 1; /* progresso garantido */
        int32_t n = bpe_encode_pretok(tk, text + pos, tok_len,
                                      out_ids + n_out, max_ids - n_out);
        if (n < 0) return -1;
        n_out += n;
        pos += tok_len;
    }
    return n_out;
}

/* Procura a próxima ocorrência de qualquer token especial a partir de
 * `from`. Retorna a posição (ou len se não houver) e o id em *out_id
 * (match mais longo em caso de empate de posição). */
static size_t find_next_special(const ov_tokenizer *tk, const char *text,
                                size_t len, size_t from, int32_t *out_id,
                                size_t *out_len) {
    size_t best_pos = len;
    int32_t best_id = -1;
    size_t best_len = 0;
    for (uint32_t i = 0; i < tk->n_special; i++) {
        int32_t id = tk->special_ids[i];
        const char *s = tk->strs[id];
        size_t slen = tk->str_lens[id];
        if (slen == 0 || slen > len - from) continue;
        const char *hit = strstr(text + from, s);
        if (!hit) continue;
        size_t pos = (size_t)(hit - text);
        if (pos < best_pos || (pos == best_pos && slen > best_len)) {
            best_pos = pos;
            best_id = id;
            best_len = slen;
        }
    }
    *out_id = best_id;
    *out_len = best_len;
    return best_pos;
}

int32_t ov_tokenizer_encode(const ov_tokenizer *tk, const char *text,
                            int32_t *out_ids, int32_t max_ids) {
    size_t len = strlen(text);
    size_t pos = 0;
    int32_t n_out = 0;

    while (pos < len) {
        int32_t sp_id;
        size_t sp_len;
        size_t sp_pos = find_next_special(tk, text, len, pos, &sp_id, &sp_len);

        if (sp_pos > pos) {
            int32_t n = encode_plain(tk, text + pos, sp_pos - pos,
                                     out_ids + n_out, max_ids - n_out);
            if (n < 0) return -1;
            n_out += n;
        }
        if (sp_id >= 0) {
            if (n_out >= max_ids) return -1;
            out_ids[n_out++] = sp_id;
            pos = sp_pos + sp_len;
        } else {
            pos = len;
        }
    }
    return n_out;
}

int32_t ov_tokenizer_special_id(const ov_tokenizer *tk, const char *literal) {
    return str_map_find(tk, literal, strlen(literal));
}

int32_t ov_tokenizer_vocab_size(const ov_tokenizer *tk) { return tk->n_vocab; }
