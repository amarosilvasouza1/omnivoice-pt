# OmniVoice-PT — pacote completo para Vulkan

Este é o pacote completo para estudar, modificar, compilar e executar o
OmniVoice-PT com o backend Vulkan. As edições e ZIPs antigos não são alterados.

## Incluído

- Todo o runtime C em `src/`.
- GGML e backend Vulkan em `third_party/ggml/`.
- Código Python original do OmniVoice em `omnivoice-src/`.
- Ferramentas Python, API, exportadores e criador de perfis em `tools/`.
- Testes e arquivos dourados de validação.
- Modelos Q4, Q8 e FP16.
- Decoder F16 e tokenizer.
- Perfis e WAVs das vozes públicas do projeto.
- Scripts de compilação Vulkan para Windows e Linux.
- Documentação e scripts auxiliares do projeto.

Não são incluídos dados privados de usuários, chaves de API, token de
administração, configuração do ngrok, bancos, áudios gerados, logs, caches,
ambientes virtuais ou pastas de compilação antigas.

O Vulkan SDK também não é redistribuído: cada desenvolvedor deve instalá-lo pelo
fornecedor oficial para aceitar a licença e receber uma versão compatível com o
sistema.

## Estrutura principal

```text
src/                    runtime C editável
third_party/ggml/       GGML e backend Vulkan
omnivoice-src/          implementação Python e treinamento
tools/                  exportação, API e clonagem de voz
tests/                  testes-fonte
build/                  modelos, decoder, tokenizer e goldens
data/voices/            vozes públicas
packaging/              launchers e documentação das outras edições
```

## Compilar no Windows

Instale:

1. Visual Studio ou Build Tools com **Desenvolvimento para desktop com C++**.
2. CMake 3.20 ou mais recente.
3. Vulkan SDK.
4. Driver atualizado da GPU com suporte Vulkan.

Abra um terminal novo e execute:

```text
COMPILAR-VULKAN-WINDOWS.bat
```

O executável ficará em `build-vulkan\bin` ou
`build-vulkan\bin\Release`. Mantenha as DLLs geradas junto dele.

## Compilar no Linux

Instale compilador C/C++, CMake, Ninja ou Make, headers/loader Vulkan e `glslc`
ou `glslangValidator`. Execute:

```bash
chmod +x compilar-vulkan-linux.sh
./compilar-vulkan-linux.sh
```

## Executar

Exemplo no Windows, a partir da raiz do pacote:

```text
build-vulkan\bin\Release\omnivoice-pt.exe --gpu -m build\omnivoice-pt-q8_0.gguf -d build\higgs-decoder-f16.gguf -k build\tokenizer.bin -p data\voices\sova.ovprompt -t "Olá, este é um teste em português." -o teste.wav
```

Se o gerador colocou o executável diretamente em `build-vulkan\bin`, ajuste
somente esse primeiro caminho.

## Qual modelo escolher

- `omnivoice-pt-q8_0.gguf`: recomendado; boa qualidade e baixo uso de memória.
- `omnivoice-pt-f16.gguf`: maior fidelidade e maior uso de VRAM.
- `omnivoice-pt-q4_0.gguf`: menor tamanho; útil em hardware limitado.

## Testes

Os arquivos dourados estão em `build/`. Para incluir os testes na compilação,
troque `-DOVPT_BUILD_TESTS=OFF` por `-DOVPT_BUILD_TESTS=ON` no script ou rode o
CMake manualmente. Depois execute `ctest` dentro de `build-vulkan`.

## API

O arquivo `tools/api_server.py` acompanha o pacote como fonte editável. A cópia
atual do servidor foi criada para a instalação original em `D:\omnivoice-pt` e
para builds CUDA; altere `ROOT` e `BUILD_DIRS` no começo do arquivo para apontar
ao diretório e ao executável Vulkan compilado pelo seu amigo.

## Licenças e responsabilidade

As licenças dos componentes estão preservadas nas respectivas pastas, incluindo
`third_party/ggml/LICENSE` e `omnivoice-src/LICENSE`.

As vozes são fornecidas para experimentação técnica. Quem redistribuir ou
publicar áudios deve confirmar que possui autorização para utilizar cada voz e
deixar claro quando o conteúdo for sintético.

