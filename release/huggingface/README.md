---
license: other
license_name: omnivoice-pt-models
license_link: LICENSE.md
language:
- pt
base_model: k2-fsa/OmniVoice
base_model_relation: quantized
pipeline_tag: text-to-speech
tags:
- gguf
- ggml
- text-to-speech
- tts
- portuguese
- voice-cloning
- omnivoice
- higgs-audio-2
---

# OmniVoice-PT (GGUF)

Modelos do **OmniVoice-PT**, um runtime em C (GGML) que roda o
[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) só para português, sem
PyTorch e sem Whisper. Código, instruções de compilação e executáveis:
**https://github.com/amarosilvasouza1/omnivoice-pt**

> **Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson
> AI USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta
> Llama 3 Community License, Copyright Meta Platforms, Inc., All Rights
> Reserved.** Built with Meta Llama 3.

> ⚠️ **Somente uso não comercial.** Os pesos do OmniVoice original são
> CC BY-NC 4.0, e estes modelos herdam essa restrição. Veja
> [Licenças](#licenças).

*English summary: Portuguese-only GGUF conversion of k2-fsa/OmniVoice for a
lightweight C/GGML runtime (no PyTorch, no Whisper). Text vocabulary pruned to
Portuguese, quantized to Q8_0/Q4_0/F16. Non-commercial use only (CC BY-NC 4.0
inherited from the original weights); the audio decoder is under the Boson
Higgs Audio 2 Community License. Clone only voices you have consent to use.*

## Arquivos

Você precisa de **um** modelo principal, do decoder e do tokenizer.

| Arquivo | O que é | Tamanho | Licença |
|---|---|---|---|
| `omnivoice-pt-q8_0.gguf` | modelo principal, Q8_0, **recomendado para GPU** | 580 MB | CC BY-NC 4.0 |
| `omnivoice-pt-f16.gguf` | modelo principal, F16, máxima qualidade | 1,1 GB | CC BY-NC 4.0 |
| `omnivoice-pt-q4_0.gguf` | modelo principal, Q4_0, o menor (bom para CPU) | 323 MB | CC BY-NC 4.0 |
| `higgs-audio-2-decoder-f16.gguf` | decoder de áudio (codec), F16 | 46 MB | Boson Higgs Audio 2 Community License |
| `tokenizer.bin` | tokenizer BPE (Qwen3 + tokens do OmniVoice) | 3,3 MB | CC BY-NC 4.0 |

## Desempenho

Medido na RTX 5060 Ti, mesma frase, 32 passos, seed 42, VRAM pelo `nvidia-smi`
nos dois lados (detalhes no `PLANO.md` do repositório):

| | PyTorch original (fp16) | Q4_0 | Q8_0 | F16 |
|---|---|---|---|---|
| VRAM pico | 2465 MB | **745 MB** | 1013 MB | 1513 MB |
| RTF (menor = mais rápido) | 0,222 | **0,157** | 0,160 | 0,162 |
| Disco total para sintetizar | 9,5 GB | **426 MB** | 683 MB | ~1,2 GB |

Q8_0, F16 e Q4_0 ficaram indistinguíveis na transcrição automática.

## Como usar

1. Compile o runtime a partir do código no
   [GitHub](https://github.com/amarosilvasouza1/omnivoice-pt). Executáveis
   prontos para Windows entram nos Releases de lá em breve.
2. Baixe os modelos:

```bash
hf download amarosilva/OmniVoice-PT-GGUF omnivoice-pt-q8_0.gguf higgs-audio-2-decoder-f16.gguf tokenizer.bin --local-dir modelos
```

3. Gere a fala:

```bash
omnivoice-pt -m modelos/omnivoice-pt-q8_0.gguf -d modelos/higgs-audio-2-decoder-f16.gguf -k modelos/tokenizer.bin -t "Olá! Este áudio foi gerado no meu computador." -o saida.wav --gpu
```

Sem `--gpu`, roda na CPU. Opções úteis: `--steps 16 --t_shift 0.3` (cerca de 2×
mais rápido, mesma inteligibilidade nos testes), `--seed N`, `--speed F`.

**Clonar uma voz:** gere um arquivo `.ovprompt` a partir de um WAV com
`tools/make_prompt.py` (precisa de Python e do OmniVoice original) e passe com
`-p voz.ovprompt`. **Nenhuma voz vem incluída neste repositório.**

## O que foi mudado em relação ao original

As licenças pedem que as mudanças fiquem claras:

- **Fonte:** `k2-fsa/OmniVoice`, revisão
  `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.
- **Modelo principal:** o vocabulário de texto foi cortado de 151.936 para
  97.542 tokens (tokens especiais, os 256 tokens de byte e os tokens de
  escrita latina/português). Como o modelo não tem cabeça de texto, isso não
  muda a saída. Depois foi convertido para GGUF e quantizado (Q8_0, Q4_0) ou
  mantido em F16.
- **Decoder:** só a parte de decodificação do tokenizer Higgs Audio 2 que vem
  em `audio_tokenizer/` (lookup RVQ + decoder DAC), convertida para GGUF F16.
  O encoder e o HuBERT foram removidos, porque só servem para criar vozes.
- **Validação:** o forward bate com o PyTorch com cosseno 1,000000 e o decoder
  com 0,99998.

Este projeto não é afiliado à k2-fsa, à Xiaomi, à Boson AI nem à Meta.

## Licenças

Este repositório tem arquivos com **licenças diferentes**. Os detalhes estão em
[LICENSE.md](LICENSE.md) e os avisos obrigatórios em [NOTICE](NOTICE).

- **Modelo principal e tokenizer: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).**
  É a licença dos pesos do OmniVoice (por causa dos dados de treino Emilia).
  Pode usar, copiar e modificar com atribuição, **mas não para fins
  comerciais**: nada de vender o áudio, cobrar por um serviço ou usar em
  produto pago.
- **Decoder (`higgs-audio-2-decoder-f16.gguf`): Boson Higgs Audio 2 Community
  License**, baseada na Meta Llama 3 Community License. O texto completo está
  em `licenses/`. Entre outras regras: siga a
  [Política de Uso Aceitável do Llama 3](https://llama.meta.com/llama3/use-policy)
  e não use as saídas para treinar outros modelos de linguagem.
- **Código do runtime:** Apache 2.0, no GitHub.

## Uso responsável

- **Clone apenas a sua voz ou a de quem autorizou.** Imitar a voz de alguém
  sem consentimento pode violar direitos de personalidade e leis de proteção
  de dados (no Brasil, a LGPD trata a voz como dado pessoal).
- É proibido usar estes modelos para fraude, golpe, se passar por outra
  pessoa, desinformação ou qualquer atividade ilegal. Os autores do OmniVoice
  proíbem expressamente a clonagem não autorizada.
- Quando publicar um áudio gerado, deixe claro que é voz sintética.
- Os modelos são fornecidos "como estão", sem garantia. Quem usa é
  responsável pelo que gera.
