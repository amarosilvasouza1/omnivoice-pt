# OmniVoice-PT

Síntese de voz (TTS) em português rodando num runtime enxuto em C/GGML,
a partir do [OmniVoice](https://github.com/k2-fsa/OmniVoice) da k2-fsa.
Sem PyTorch, sem Whisper: só o que é preciso para falar português.

| | PyTorch original (fp16) | OmniVoice-PT Q8_0 | OmniVoice-PT Q4_0 |
|---|---|---|---|
| VRAM pico | 2465 MB | 1013 MB | **745 MB** |
| VRAM pico clonando voz | 4356 MB | **1565 MB** | — |
| RTF (menor = mais rápido) | 0,222 | 0,160 | **0,157** |
| Disco total | 9,5 GB | 683 MB | **426 MB** |

Medido na RTX 5060 Ti. Roda também em CPU (inclusive em Intel de 2ª geração)
e em placas AMD via Vulkan. Detalhes, metodologia e validação em
[PLANO.md](PLANO.md).

> ⚠️ **Os modelos são para uso não comercial** (CC BY-NC 4.0, herdado do
> OmniVoice). O código deste repositório é Apache 2.0. Veja [Licenças](#licenças).

## Baixar os modelos

Os modelos ficam no Hugging Face: **https://huggingface.co/amarosilva/OmniVoice-PT-GGUF**

```bash
hf download amarosilva/OmniVoice-PT-GGUF omnivoice-pt-q8_0.gguf higgs-audio-2-decoder-f16.gguf tokenizer.bin --local-dir modelos
```

Use `omnivoice-pt-q8_0.gguf` na GPU, `omnivoice-pt-q4_0.gguf` na CPU ou
`omnivoice-pt-f16.gguf` para a máxima qualidade.

## Compilar

Requisitos: CMake 3.20+, um compilador C (MSVC no Windows, gcc/clang no Linux)
e, para GPU NVIDIA, o CUDA Toolkit 12.

```bash
git clone --recursive https://github.com/amarosilvasouza1/omnivoice-pt.git
```

GPU NVIDIA (no Windows, rode num "x64 Native Tools Command Prompt"):

```bash
cmake -S . -B cmake-build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON
```

```bash
cmake --build cmake-build-cuda --target omnivoice-pt
```

- Só CPU: tire o `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON`.
- AMD/Intel via Vulkan: troque por `-DGGML_VULKAN=ON`.
- CPU antiga sem AVX2 (Linux): `cmake --preset linux-q4-sandybridge` e depois
  `cmake --build --preset linux-q4-sandybridge`.

O executável sai em `cmake-build-cuda/bin/`.

## Usar

```bash
omnivoice-pt -m modelos/omnivoice-pt-q8_0.gguf -d modelos/higgs-audio-2-decoder-f16.gguf -k modelos/tokenizer.bin -t "Olá! Este áudio foi gerado no meu computador." -o saida.wav --gpu
```

| Opção | O que faz |
|---|---|
| `--gpu` | usa a GPU (sem ela, roda na CPU) |
| `--steps N` | passos de difusão (padrão 32). `--steps 16 --t_shift 0.3` é cerca de 2× mais rápido |
| `--seed N` | semente; a mesma semente gera o mesmo áudio |
| `--speed F` | velocidade da fala |
| `-p voz.ovprompt` | clona uma voz (veja abaixo) |
| `--serve` | modo worker: mantém o modelo carregado e lê um pedido por linha |

### Clonar uma voz

`tools/make_prompt.py` transforma um WAV de referência num `.ovprompt` de
poucos KB. Ele precisa de Python e do OmniVoice original (submódulo
`omnivoice-src`), mas só roda uma vez por voz: o runtime C nunca carrega o
Whisper nem o encoder.

**Clone apenas a sua voz ou a de quem autorizou.** Veja
[Uso responsável](#uso-responsável).

### Servidor para compartilhar com amigos

`tools/api_server.py` sobe uma API HTTP com painel de administração, uma chave
por pessoa e limites por conexão. Ele procura os arquivos em `build/` com os
nomes `omnivoice-pt-q8_0.gguf`, `higgs-decoder-f16.gguf` e `tokenizer.bin`
(renomeie o decoder baixado). Se for expor na internet, mantenha o código
atualizado.

## Reproduzir os modelos

Os modelos publicados saem destes scripts, a partir da revisão
`c5fdb5ccb189668d56333f77ba2629f4cd7535f4` do `k2-fsa/OmniVoice`:

```bash
python tools/export.py --quant Q8_0
```

```bash
python tools/export_decoder.py
```

O `export_decoder.py` grava o decoder como `build/higgs-decoder-f16.gguf`, que
é o nome usado pelo código e pelos scripts. O nome
`higgs-audio-2-decoder-f16.gguf` é só o do arquivo publicado, por exigência da
licença da Boson AI. O conteúdo é o mesmo.

## Licenças

- **Código deste repositório:** [Apache 2.0](LICENSE).
- **Modelos principais e tokenizer:** CC BY-NC 4.0, herdada dos pesos do
  OmniVoice. **Não pode usar para fins comerciais.**
- **Decoder:** Boson Higgs Audio 2 Community License
  ([texto](licenses/BOSON-HIGGS-AUDIO-2-COMMUNITY-LICENSE.txt)), que incorpora
  a Meta Llama 3 Community License.
- **ggml:** MIT ([texto](licenses/GGML-MIT-LICENSE.txt)).

Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI
USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta Llama 3
Community License, Copyright Meta Platforms, Inc., All Rights Reserved.
Built with Meta Llama 3.

Os avisos completos estão em [NOTICE](NOTICE). Este projeto não é afiliado à
k2-fsa, à Xiaomi, à Boson AI nem à Meta.

## Uso responsável

- **Clone apenas a sua voz ou a de quem autorizou.** Imitar a voz de alguém
  sem consentimento pode violar direitos de personalidade e a LGPD, que trata
  a voz como dado pessoal.
- Não use para fraude, golpe, se passar por outra pessoa ou desinformação.
- Quando publicar um áudio gerado, deixe claro que é voz sintética.
- Nenhuma voz de pessoa real vem neste repositório.
