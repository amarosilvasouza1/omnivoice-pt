# OmniVoice-PT: runtime enxuto em C para TTS em português

Objetivo: pegar o [OmniVoice (k2-fsa)](https://github.com/k2-fsa/OmniVoice) e reduzir o consumo
de VRAM de ~5–7 GB para **menos de 1 GB**, ganhando velocidade, sem perder qualidade —
carregando somente o que é necessário para falar português. Máquina alvo: GPU com < 3 GB de VRAM
(fallback CPU obrigatório).

Fonte dissecada: `omnivoice-src/` (clone raso de k2-fsa/OmniVoice, jul/2026).

---

## 1. Arquitetura real (mapeada do código)

Pipeline em `omnivoice/models/omnivoice.py`:

```
texto ─┐
       ├─► tokenizer BPE (Qwen3, vocab 151.936)
prefixo de estilo: <|denoise|><|lang_start|>pt<|lang_end|><|instruct_start|>None<|instruct_end|>
       │
       ▼
Qwen3-0.6B (AutoModel, SEM lm_head) — atenção BIDIRECIONAL (não-causal, não-autoregressivo)
  entrada  = embeddings de texto  OU  soma dos 8 audio_embeddings (posições de áudio)
  saída    = audio_heads: Linear(1024 → 8×1025)  →  logits [B, 8, T, 1025]
       │
       ▼
Difusão mascarada: 32 passos (num_step), CFG guidance_scale=2.0 (batch 2B: cond+uncond),
  a cada passo desamascara top-k posições por confiança (penalidade por camada de codebook,
  gumbel na posição). Sequência toda re-processada a cada passo (sem KV cache).
       │
       ▼
HiggsAudioV2Tokenizer.decode()  (RVQ 8 codebooks, 1025 vocab, frame_rate ~25Hz) → áudio 24 kHz
```

Componentes carregados pelo `from_pretrained()` hoje:

| Componente | Tamanho no disco | Uso real |
|---|---|---|
| `model.safetensors` (Qwen3-0.6B + audio_embeddings + audio_heads) | 2.45 GB **em F32** | necessário (inferência roda em fp16) |
| `audio_tokenizer/` (HiggsAudioV2, encoder+decoder) | ~0.8 GB | **só o decoder é necessário no runtime** |
| Whisper large-v3-turbo | ~1.6 GB | **só transcreve o áudio de referência** — evitável 100% |
| PyTorch + contexto CUDA | ~1–1.5 GB | eliminado pelo GGML |

Fatos que destravam a otimização:

1. **`VoiceClonePrompt` é minúsculo e serializável**: `ref_audio_tokens` (int 8×T, ~poucos KB),
   `ref_text` (string), `ref_rms` (float). Pré-computa uma vez em Python (com Whisper e o
   encoder Higgs) e salva em binário. O runtime C nunca carrega Whisper nem o encoder.
2. **Não existe cabeça de texto**: a saída é só `audio_heads` (8×1025×1024 ≈ 8.4M params).
   A matriz de embeddings de texto (151.936 × 1024 ≈ **155M params = 620 MB f32**) é usada
   apenas como entrada ⇒ **podável para o subconjunto PT sem nenhum efeito colateral**.
3. **Idioma é só prompt**: `<|lang_start|>pt<|lang_end|>` — nada de módulos por idioma.
   Os 600+ idiomas compartilham pesos; o corte real é no vocabulário.
4. **`audio_embeddings`**: Embedding(8×1025, 1024) ≈ 8.4M params — barato, mantém inteiro.
5. **Duration estimator é baseado em regras** (`RuleDurationEstimator`) — porta direto pra C.
6. **O tokenizer de áudio Higgs V2 embute um HuBERT (~95M params) usado só no encode**
   (destilação semântica) — descartado no runtime junto com o encoder DAC. O decode precisa
   apenas de: lookup RVQ (codebook_dim 64) + decoder DAC (hidden 1024). hop 960 @ 24 kHz
   ⇒ frame_rate 25 Hz.
7. Ineficiências do pipeline PyTorch que o runtime C corrige:
   - batch uncond é *padded* até o comprimento do cond (compute desperdiçado × 32 passos);
   - máscara de atenção materializada como bool `[2B, 1, L, L]` (O(L²) de memória);
   - logits promovidos a f32 para a sequência inteira a cada passo — só as posições ainda
     mascaradas importam;
   - CFG recalcula log_softmax duplo — dá para fundir num kernel só.

## 2. Orçamento de VRAM alvo

| Componente (runtime C) | Q8_0 | Q4_K |
|---|---|---|
| Backbone Qwen3-0.6B (sem embeddings de texto) | ~470 MB | ~260 MB |
| Embeddings de texto podados p/ PT (~32k tokens, fp16) | ~65 MB | ~65 MB |
| audio_embeddings + audio_heads (fp16) | ~34 MB | ~34 MB |
| Decoder Higgs (fp16) | ~200–400 MB | ~200–400 MB |
| Ativações (seq ~1.5k tokens, batch 2) | ~100 MB | ~100 MB |
| **Total** | **~0.9 GB** | **~0.7 GB** |

Sem PyTorch, sem Whisper, sem encoder de áudio, sem 600 idiomas de vocabulário.

## 3. Fases

### Fase 1 — Dissecação ✅
Concluída; este documento é o resultado.

### Fase 2 — Exportador (Python, roda uma vez)
`tools/export.py`:
1. Baixa `k2-fsa/OmniVoice` do HF.
2. **Poda PT**: roda um corpus PT-BR (ex.: wikipedia-pt + frases coloquiais) pelo tokenizer,
   coleta tokens usados + tokens especiais (`<|lang_start|>`, tags não-verbais `[laughter]` etc.),
   monta tabela de remapeamento id_antigo→id_novo e fatia a matriz de embeddings.
3. Converte para **GGUF**: backbone Q8_0/Q4_K, embeddings fp16, audio_heads fp16.
   Metadados: remapeamento de vocab, config (8 codebooks, 1025 vocab, mask_id 1024, frame_rate).
4. Exporta o **decoder** do HiggsAudioV2Tokenizer (fp16) — GGUF próprio ou ONNX (fallback).
5. `tools/make_prompt.py`: dado um WAV de referência, roda Whisper + encoder Higgs UMA vez e
   salva `voz.ovprompt` (header + ref_text + ref_rms + tokens int16 8×T).

### Fase 3 — Runtime C (GGML)
`src/`:
- `tokenizer.c` — BPE do Qwen3 sobre o vocab podado (tabela no GGUF), tags não-verbais.
- `model.c` — grafo Qwen3 em GGML: RoPE, GQA **bidirecional**, RMSNorm, SwiGLU;
  entrada mista texto/áudio (soma dos 8 embeddings de codebook com offsets — igual a
  `_prepare_embed_inputs`).
- `sampler.c` — cronograma de 32 passos com `t_shift`, CFG fundido, penalidade de camada,
  gumbel por posição, unmask top-k. Cond e uncond com comprimentos reais (sem padding).
- `decoder.c` — decoder Higgs (convs/upsampling) → PCM 24 kHz.
- `main.c` — CLI: `omnivoice-pt -m modelo.gguf -p voz.ovprompt -t "texto" -o out.wav`.
- Backends GGML: CUDA (GPU <3 GB) e CPU (AVX2). Pesos via mmap.

### Fase 4 — Validação
- Áudio A/B vs pipeline PyTorch original (mesmo prompt/config).
- Medir: VRAM real (nvidia-smi), RTF, WER com um ASR PT.
- Explorar: `num_step` 16–24 (menos passos = quase 2× mais rápido), quantização por camada
  (embeddings/heads mais sensíveis), fp16 vs bf16 nas ativações.

## 4. Resultados (10/07/2026)

**Funciona de ponta a ponta.** `omnivoice-pt.exe` sintetiza fala PT inteligível
(transcrição Whisper bate palavra por palavra com o texto de entrada).

| Métrica | Original (PyTorch) | Runtime C (Q8_0, RTX 5060 Ti) |
|---|---|---|
| VRAM | ~5–7 GB | **609 MB pesos + 20 MB ativações (~0.9 GB c/ contexto CUDA)** |
| Disco (modelo) | 3.27 GB + Whisper 1.6 GB | 609 MB + decoder 48 MB + tokenizer 13 MB |
| RTF | — | **0.51–0.64** (2× mais rápido que tempo real) |

Validações (todas automatizadas em `tests/`):
- Tokenizer C = HuggingFace bit a bit (13 casos, incl. acentos, especiais, tags).
- Forward GGML = PyTorch f32: cosseno 1.000000, argmax 80/80 (F16/CPU).
- Decoder DAC C = PyTorch: cosseno 0.999999, erro máx 0.001.
- Áudio final transcrito pelo Whisper com 100% de acerto.

Componentes eliminados do runtime: PyTorch (~1.5 GB), Whisper (1.6 GB),
encoder DAC + HuBERT (~180M params), 54k embeddings de texto (~35%),
armazenamento F32 (4x maior que Q8).

### Otimização de velocidade (11/07/2026)

Alvo: deixar o runtime mais rápido sem perder qualidade. RTF caiu de
**~0.41 → ~0.24** (≈40% mais rápido) para a mesma frase de ~6.6s, na RTX 5060 Ti:

| Etapa | Antes | Depois | Como |
|---|---|---|---|
| Decode DAC | 1.1s | **0.2s** | movido para a GPU (`ov_decoder_load(path, use_gpu)`); kernel da conv transposta convertido para F32 no grafo (CUDA exige F32, CPU exige F16) |
| Geração (32 passos) | 1.7s | **1.4s** | grafos persistentes por forma (cond/uncond) + `GGML_CUDA_GRAPHS=ON` → replay de CUDA Graph em vez de relançar ~27k kernels |
| Sampler | — | menor | pula log_softmax/CFG de posições já desmascaradas |

Corretude preservada: golden do forward cosseno 0.999999, decoder 0.999983 (GPU),
transcrição Whisper 100%. `GGML_CUDA_GRAPHS` religado via `reconfig_cuda.bat`
(vem OFF por padrão no ggml).

Medição por passos (mesma frase, seed fixa) — inteligibilidade (WER 0%) se mantém
até 12 passos; naturalidade a validar de ouvido. Default segue em 32 (via `--steps`).

### Qualidade (11/07/2026)

- Sampler skip agora **preserva o stream do RNG** (avança gumbel também nas posições
  puladas) → saída bit-idêntica à versão pré-otimização; remove qualquer regressão do
  ganho de velocidade.
- Q8 vs F16 é diferença **perceptual** (ambos inteligíveis; formas de onda divergem por
  a difusão ser caótica, não dá para medir por correlação). Web tool ganhou seletores de
  **quant (Q8/F16)** e **passos (16–64)** para o usuário calibrar de ouvido. F16 = ~1.1 GB
  (ainda < PyTorch 2.1 GB), quase sem custo de velocidade.

Pendências:
- [ ] Fundir cond+uncond num forward batch-2 (melhor ocupância da GPU; exige máscara
      de atenção para o padding do uncond) — maior ganho restante na geração.
- [ ] Chunking para textos > 30s (hoje trunca em 750 frames).
- [x] **Clonagem de voz validada e modularizada** (11/07): `make_prompt.py` cria
      `.ovprompt` de um WAV real; `omnivoice-pt -p voz.ovprompt` clona num texto novo,
      transcrição 100%. O `.ovprompt` É o módulo: só carrega sob demanda (`-p`), e clonar
      adiciona apenas ~8–28 MB de VRAM sobre a geração normal (a referência é minúscula).
      **`--max_ref_seconds` corta a referência** (ex. 6s): clonagem ~2× mais rápida
      (gen 4.1s→2.6s) com fidelidade equivalente, pois a sequência de atenção encurta.
      Web tool: voz Automática / Clone rápido (6s) / Clone fidelidade (14s), mesma
      referência nos dois pipelines (loader `.ovprompt`→VoiceClonePrompt).
- [ ] Pós-processamento: remoção de silêncio, fade (hoje só normalização).
- [ ] Q4_K para o backbone (~350 MB de pesos).
- [ ] A/B formal vs pipeline PyTorch com o mesmo prompt (WER/MOS).

### Variante Linux Q4 para CPU antiga (11/07/2026)

- Pacote separado `OmniVoice-PT-Q4-Linux.zip`; a versão Windows e os
  modelos antigos permanecem intactos.
- Binário Linux estático (musl), sem dependências de runtime, compilado para
  Sandy Bridge com SSE4.2 + AVX e `-mtune=sandybridge`; AVX2, FMA, F16C, BMI2,
  CUDA, BLAS e OpenMP ficam desativados para evitar `Illegal instruction` no
  Core i7 de 2ª geração.
- Corrigida a portabilidade de `_fseeki64`, a detecção de CPUs Linux e a
  medição de tempo (agora monotônica e real, em vez de tempo de CPU agregado).
- Launcher `gerar.sh`: Q4_0, 8 passos por padrão e número de threads igual aos
  núcleos físicos; `OV_STEPS` e `OV_THREADS` permitem ajuste manual.

## API pública + painel de admin (11/07/2026)

`tools/api_server.py` (porta 5180) — serviço para compartilhar o runtime C
com outras pessoas na rede, com controle total pelo dono:

- **Cada amigo = uma "conexão"** com chave própria (`ov_...`), criada/removida
  pelo admin. Link pronto para compartilhar: `/app?key=CHAVE` (página simples,
  sem precisar de código) ou uso direto via `POST /api/v1/generate`.
- **Configurável por conexão**: máx. caracteres, passos máximos, quant.
  permitidas (Q8/F16), vozes permitidas, limite de pedidos/min. Ativar/
  desativar sem apagar.
- **Painel `/admin`** (login por token, gerado em `build/admin_token.txt`):
  estatísticas globais, tabela de conexões com config inline, e **atividade
  ao vivo** (atualiza a 3s) — texto pedido, áudio tocável, tempo de geração,
  RTF, VRAM, por conexão.
- Banco: SQLite em `build/api.db` (tabelas `connections` e `requests`).
  Rate limit em memória (janela deslizante de 60s por chave).
- Só expõe o **runtime C** (leve) — não carrega o PyTorch original, de
  propósito, para o serviço ficar barato de manter no ar.
- Acesso: LAN via IP da máquina (`IP-DA-MAQUINA:5180`); para
  internet, precisaria de port-forward no roteador ou um túnel
  (ngrok/cloudflared) — não configurado automaticamente.

Bug corrigido durante o teste: ordem de `name`/`api_key` trocada no INSERT
da conexão padrão (values na ordem errada) — teste ponta a ponta pegou na hora.

**Documentação pública em `/docs`** (sem precisar de chave): explica uso simples
(link `/app?key=`), referência completa da API (`POST /generate`, `GET /voices`,
`GET /audio/<f>`, exemplos curl/Python com botão copiar), tabela de limites e
de códigos de erro. URL base é preenchida dinamicamente (`request.host_url`),
então funciona certo tanto em `localhost` quanto no IP da LAN.

### Sampler paralelo + knobs de qualidade expostos (25/07/2026)

**1) O sampler era ~30% do tempo de geração, em uma única thread.** Entre os dois
forwards da GPU, cada passo fazia 3 `log_softmax` por posição (`double exp`/`log`),
em série. Medido no CPU desta máquina, T=158 (1264 posições, V=1025):

| | ms/passo |
|---|---|
| serial (como estava) | 14,8 |
| com OpenMP | **1,9** (7,8×) — saídas diferentes: **0/1264** |

Como o cronograma é bem carregado no fim (`t_shift=0.1` ⇒ em média **83%** das
posições seguem mascaradas), esse custo era quase máximo em todo passo. Ganho
ponta a ponta, frase de 6,3 s, Q8_0, RTX 5060 Ti, 32 passos:

| | geração | RTF |
|---|---|---|
| antes | 1,29 s | 0,24 |
| depois | **0,96 s** (mediana de 5; mín. 0,92) | **0,19** |

**Saída bit-idêntica** (MD5 igual ao do binário anterior) — o gumbel de posição é
sorteado numa pré-passada serial, mantendo a ordem e a contagem de sorteios do
laço original; o RNG fica desacoplado do laço de trabalho, que é paralelo por
posição (cada posição só toca sua própria linha de logits). Os 3 goldens passam.
OpenMP é opcional (`OVPT_OPENMP`, ligado por padrão) e fica fora do binário
estático do Linux, que segue sem OpenMP de propósito. `--threads N` também
governa o sampler.

**2) `t_shift`, `guidance_scale` e `class_temperature` agora existem na CLI**
(`--t_shift`, `--cfg`, `--temp`) — antes eram hardcoded, o que impedia testar a
fronteira qualidade/velocidade sem recompilar. `class_temperature` não estava
implementado no C (só greedy); agora segue o original (`_filter_top_k(ratio=0.1)`
+ gumbel, com a confiança ainda vindo do máximo da linha). O RNG do sorteio por
classe é derivado de `(seed, passo, posição)`, então independe da ordem das
threads — verificado: duas rodadas com a mesma seed dão o mesmo MD5. No modo
`--serve` os knobs valem para o worker todo (o protocolo não mudou; steps e seed
seguem por job).

Sweep inicial (mesma frase, transcrição Whisper palavra por palavra):

| config | geração | transcrição |
|---|---|---|
| 32 passos (padrão) | 0,92 s | 100% |
| 24 passos, `--t_shift 0.2` | 0,72 s | 100% |
| 16 passos, `--t_shift 0.3` | 0,53 s | 100% |
| 16 passos, `t_shift` padrão | 0,52 s | 100% |
| 32 passos, `--temp 0.7` | 0,97 s | 100% |
| 32 passos, `--cfg 0` | 0,53 s | **falha** ("runtime em" → "Hantamin") |

Motivo de mexer em `t_shift` junto com os passos: com 32 passos, 22% dos tokens
são decididos no último passo; com 12 passos são **47%** (sem refino nenhum).
`--steps 16 --t_shift 0.3` volta a 17% — perfil melhor que o padrão atual pela
metade do custo. Inteligibilidade se mantém em todas as variantes; **falta o
A/B de naturalidade de ouvido** para escolher o novo padrão.

`--cfg 0` fica 42% mais rápido mas quebra a aderência ao texto — CFG não é
descartável. O caminho nuançado seria aplicar CFG só num intervalo de passos.

Pendências desta rodada:
- [ ] Escolher o padrão de `steps`/`t_shift` de ouvido (candidato: 16 / 0.3).
- [ ] Passar os knobs pelo painel/API (hoje só por CLI do worker).
- [ ] Rebuild dos outros alvos (`cmake-build-cuda-*`, `-vulkan`, Linux Q4) e do
      `dist/` — só o `cmake-build-cuda` foi recompilado; os workers em execução
      precisam de restart para pegar o binário novo.
- [ ] Próximos do mesmo caminho: matar o transpose de ~5 MB de logits por forward
      (`ov_model.c`, o sampler pode indexar o layout cru) e reduzir os 3
      `log_softmax` a 1 — `(1+g)·lsm(c) − g·lsm(u)` difere de `(1+g)·c − g·u` por
      uma constante por linha, que **cancela** no `log_softmax` final (medido:
      `|Δscore| ≤ 7,6e-6`, ~2,6× mais barato).
- [ ] Sobre fundir cond+uncond: em vez do batch-2 padeado (aqui seria 158→193,
      +22% de compute jogado fora), empacotar os dois numa sequência única com
      máscara bloco-diagonal — zero padding, um replay de CUDA graph, uma cópia
      D2H. É o que o original faz com `flex_attention` + document ids.

### Medição A/B formal contra o PyTorch original (25/07/2026)

Fecha a pendência "A/B formal vs pipeline PyTorch". **Mesmo texto** (168 chars),
mesma GPU (RTX 5060 Ti), 32 passos, seed 42. VRAM medida pelo **mesmo método nos
dois lados** (`nvidia-smi` amostrado a 20 Hz, corrigido por baseline), porque
`torch.cuda.max_memory_allocated` não conta o contexto CUDA e não seria comparável.
Runtime C medido em `--serve` (mediana de 3 rodadas quentes, após 1 de aquecimento).

**Voz automática (sem clonagem)**

| | PyTorch fp16 | C q4_0 | C q8_0 | C f16 |
|---|---|---|---|---|
| VRAM após carga | 2131 MB | **473 MB** | 741 MB | 1223 MB |
| VRAM pico | 2465 MB | **745 MB** | 1013 MB | 1513 MB |
| geração | 2,289 s | 1,274 s | 1,313 s | 1,312 s |
| decode | (dentro do generate) | 0,314 s | 0,305 s | 0,329 s |
| total | 2,289 s | **1,588 s** | 1,618 s | 1,641 s |
| áudio gerado | 10,32 s | 10,12 s | 10,12 s | 10,12 s |
| RTF | 0,222 | **0,157** | 0,160 | 0,162 |

**Clonagem de voz** (referência de 21,2 s). O original recebe o WAV e
roda Whisper + encoder na hora; o runtime C recebe o `.ovprompt` pré-computado.

| | PyTorch fp16 + ref_audio | C q8_0 + .ovprompt |
|---|---|---|
| VRAM após carga | 1951 MB | 1032 MB |
| VRAM pico | 4356 MB | **1565 MB** |
| geração | 7,149 s | 4,037 s |
| total | 7,149 s | **4,793 s** |
| áudio gerado | 17,25 s | 23,56 s |
| RTF | 0,414 | **0,203** |

O pico do original sobe de 2465 → 4356 MB porque o Whisper large-v3-turbo entra
na memória durante o `generate()`. É exatamente o custo que o `.ovprompt` elimina.

**Disco (mínimo para sintetizar uma frase)**

| | Original | Runtime C |
|---|---|---|
| modelo | 3116 MB | 580 MB (q8) · 323 (q4) · 1063 (f16) |
| Whisper large-v3-turbo | 1547 MB | — |
| decoder + tokenizer | (dentro do modelo) | 46 + 3 MB |
| runtime (torch+transformers / exe+DLLs) | 4884 MB | 54 MB |
| **total** | **9548 MB** | **683 MB** (q8) · **426 MB** (q4) |

**Correção de um número deste documento:** a tabela da seção 4 diz "VRAM ~5–7 GB"
para o original. Medido agora, em fp16 e voz automática, são **2,1–2,5 GB**; com
clonagem via `ref_audio`, **4,4 GB**. Os 5–7 GB correspondem a outra configuração
(pesos em F32, que é como o `model.safetensors` está no disco). O ganho de VRAM
real é de **2,4× (auto)** a **2,8× (clonagem)**, não de 5–7×. O ganho de **disco**
é que é grande: **14×**.

**O que o trabalho de hoje mudou** (mesmo texto, CLI, mediana de 3; `OMP_NUM_THREADS=1`
reproduz o sampler serial de antes):

| | geração | decode | total | RTF |
|---|---|---|---|---|
| C q8_0 **antes** (sampler serial) | 1,950 s | 0,350 s | 2,300 s | 0,227 |
| C q8_0 **agora** | 1,380 s | 0,330 s | **1,710 s** | **0,169** |
| PyTorch original | 2,289 s | — | 2,289 s | 0,222 |

Ou seja: **antes de hoje o runtime C empatava em velocidade com o PyTorch**
(2,300 s vs 2,289 s) — o ganho era só de VRAM e disco. O sampler paralelo é o que
tornou o runtime efetivamente mais rápido: **1,34× vs o original**, 29% vs ele mesmo.

**Qualidade** (Whisper-small, mesma frase): as 6 saídas transcrevem o mesmo
conteúdo, com os mesmos artefatos em "comparacao"/"geracao" — o texto do teste foi
escrito **sem acentos**, e o modelo pronuncia literalmente; afeta todas as linhas
igualmente. Q8, F16 e Q4_0 ficaram indistinguíveis na transcrição.

**Achado novo: clonagem infla a duração.** A mesma frase que dura 10,12 s em voz
automática sai com **23,56 s** no runtime C com clonagem e **17,25 s** no original.
Medi o silêncio de cauda: **0,00 s e 0,13 s** — não é silêncio no fim, é fala
arrastada mesmo. `estimate_target_frames` escala a duração pela razão
`ref_frames / peso(ref_text)` da referência, e uma referência de 21,2 s com
transcrição curta infla o alvo. Os dois pipelines erram, o C erra 37% mais.
Isto reforça a pendência do estimador de duração: hoje ele é o maior problema de
qualidade percebida em voz clonada, e um alvo menor também seria mais rápido.

## 5. Riscos e decisões em aberto

- **Decoder Higgs em GGML é a parte mais trabalhosa** (arquitetura de codec neural, convs
  transpostas). Fallback pragmático: decoder via onnxruntime (C API) enquanto o port GGML
  não fica pronto — ainda elimina PyTorch/Whisper e mantém o orçamento de VRAM.
- **Poda de vocab**: cobertura do corpus PT precisa ser generosa (números, siglas, anglicismos,
  emojis comuns). Token fora do vocab podado no runtime → fallback para bytes (BPE do Qwen é
  byte-level, sempre há caminho).
- Voice design (`instruct`) em EN/ZH pode ser mantido — são só tokens de texto.
- Licença Apache 2.0 — sem impedimento.
