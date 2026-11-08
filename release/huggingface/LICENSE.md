# Licenças dos arquivos

Cada arquivo deste repositório tem a licença do material de onde ele veio.

| Arquivo | Licença | Texto |
|---|---|---|
| `omnivoice-pt-q8_0.gguf`, `omnivoice-pt-f16.gguf`, `omnivoice-pt-q4_0.gguf` | Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) | https://creativecommons.org/licenses/by-nc/4.0/legalcode |
| `tokenizer.bin` | CC BY-NC 4.0 | https://creativecommons.org/licenses/by-nc/4.0/legalcode |
| `higgs-audio-2-decoder-f16.gguf` | Boson Higgs Audio 2 Community License (incorpora a Meta Llama 3 Community License) | `licenses/BOSON-HIGGS-AUDIO-2-COMMUNITY-LICENSE.txt`, `licenses/META-LLAMA-3-COMMUNITY-LICENSE.txt` |

## Atribuição

- **OmniVoice**, de k2-fsa (Xiaomi Corp.): https://huggingface.co/k2-fsa/OmniVoice,
  revisão `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`. Pesos sob CC BY-NC 4.0.
- **Qwen3-0.6B**, da Alibaba Cloud, modelo base do OmniVoice: Apache 2.0.
- **Higgs Audio 2**, da Boson AI USA, Inc.: Boson Higgs Audio 2 Community License.
- **Meta Llama 3**, da Meta Platforms, Inc.: Meta Llama 3 Community License.

## Mudanças

Os arquivos `omnivoice-pt-*.gguf` e `tokenizer.bin` são adaptações dos pesos do
OmniVoice: vocabulário de texto cortado para português (97.542 de 151.936
tokens), conversão para GGUF e quantização. O arquivo
`higgs-audio-2-decoder-f16.gguf` é o decoder do tokenizer Higgs Audio 2
convertido para GGUF F16, sem o encoder e sem o HuBERT.

## Avisos obrigatórios

Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI
USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta Llama 3
Community License, Copyright Meta Platforms, Inc., All Rights Reserved.

Built with Meta Llama 3.

Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright ©
Meta Platforms, Inc. All Rights Reserved.

Boson Higgs Audio 2 is licensed under the Boson Community License, Copyright ©
Boson AI USA, Inc. All Rights Reserved.

Nenhuma licença de marca é concedida. Este projeto não é afiliado à k2-fsa, à
Xiaomi, à Boson AI, à Meta nem à Alibaba.
