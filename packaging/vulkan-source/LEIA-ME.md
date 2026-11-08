# OmniVoice-PT — pacote de código-fonte Vulkan

Este pacote contém o código-fonte necessário para estudar, modificar e compilar
o runtime do OmniVoice-PT com o backend Vulkan. Ele não substitui nenhuma edição
pronta do projeto.

## O que está incluído

- `src/`: runtime em C, tokenizer, modelo, sampler, decoder e programa principal.
- `third_party/ggml/`: GGML e o backend Vulkan usado pelo runtime.
- `tools/`: exportadores e utilitários Python do projeto.
- `tests/`: testes do tokenizer, modelo e decoder.
- `omnivoice-src/`: código Python do projeto OmniVoice usado como referência,
  para criação de perfis de voz e para treinamento/ajuste fino.
- `CMakeLists.txt` e `CMakePresets.json`: configuração do projeto.

Modelos `.gguf`, perfis de voz, áudios, bancos de dados, chaves, tokens,
executáveis, DLLs e logs não estão incluídos. Isso mantém o pacote pequeno e
evita compartilhar dados privados.

## Compilar no Windows com Vulkan

Requisitos:

1. Windows 10 ou 11 de 64 bits.
2. Visual Studio ou Build Tools com o componente **Desenvolvimento para desktop
   com C++**.
3. CMake 3.20 ou mais recente.
4. Vulkan SDK instalado. Depois da instalação, abra um terminal novo e confira
   se a variável `VULKAN_SDK` existe.
5. Driver de vídeo atualizado com suporte a Vulkan.

Execute:

```text
COMPILAR-VULKAN-WINDOWS.bat
```

O executável será criado em uma destas pastas, dependendo do gerador instalado:

```text
build-vulkan\bin\omnivoice-pt.exe
build-vulkan\bin\Release\omnivoice-pt.exe
```

Mantenha as DLLs produzidas na mesma pasta do executável.

## Compilar no Linux com Vulkan

Instale um compilador C/C++, CMake, Ninja ou Make, os headers/loader do Vulkan e
o compilador de shaders `glslc`. Depois execute:

```bash
chmod +x compilar-vulkan-linux.sh
./compilar-vulkan-linux.sh
```

O binário ficará em `build-vulkan/bin/omnivoice-pt`.

## Arquivos necessários para executar

Compilar gera somente o programa. Para sintetizar áudio também são necessários:

- um modelo `omnivoice-pt-*.gguf`;
- `higgs-decoder-f16.gguf`;
- `tokenizer.bin`;
- opcionalmente um perfil de voz `.ovprompt`.

Use apenas modelos e vozes que você tenha permissão para copiar. Exemplo:

```text
omnivoice-pt.exe --gpu -m omnivoice-pt-q8_0.gguf -d higgs-decoder-f16.gguf -k tokenizer.bin -p voz.ovprompt -t "Olá, este é um teste em português." -o teste.wav
```

## Onde modificar

- `src/main.c`: CLI, texto, duração, carregamento e modo servidor.
- `src/ov_model.c`: grafo do modelo e backend de GPU.
- `src/ov_sampler.c`: passos de difusão, CFG e seleção de tokens.
- `src/ov_decoder.c`: conversão dos tokens de áudio para PCM.
- `src/ov_tokenizer.c`: tokenizer BPE.
- `tools/export.py`: exportação e quantização Q8, Q4 e FP16.
- `tools/make_prompt.py`: criação de perfis `.ovprompt`.

Os testes são desativados na compilação rápida porque os arquivos dourados e os
modelos grandes não acompanham este ZIP. Eles podem ser ativados com
`-DOVPT_BUILD_TESTS=ON` depois que esses artefatos forem adicionados.

## Licenças e uso responsável

As licenças dos projetos incluídos permanecem dentro das respectivas pastas,
incluindo `third_party/ggml/LICENSE` e `omnivoice-src/LICENSE`. Preserve os
avisos e licenças ao redistribuir modificações.

Use somente vozes com autorização e deixe claro quando um áudio for sintético.

