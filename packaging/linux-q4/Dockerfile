FROM alpine:3.20 AS build

RUN apk add --no-cache build-base cmake
WORKDIR /src

COPY CMakeLists.txt CMakePresets.json ./
COPY src ./src
COPY tests ./tests
COPY third_party/ggml ./third_party/ggml

RUN cmake --preset linux-q4-sandybridge -DOVPT_FULLY_STATIC=ON \
 && cmake --build --preset linux-q4-sandybridge \
 && strip cmake-build-linux-q4/bin/omnivoice-pt

FROM alpine:3.20 AS package
RUN mkdir -p /package/OmniVoice-PT-Q4-Linux/modelos
COPY --from=build /src/cmake-build-linux-q4/bin/omnivoice-pt /package/OmniVoice-PT-Q4-Linux/omnivoice-pt
COPY build/omnivoice-pt-q4_0.gguf /package/OmniVoice-PT-Q4-Linux/modelos/
COPY build/higgs-decoder-f16.gguf /package/OmniVoice-PT-Q4-Linux/modelos/
COPY build/tokenizer.bin /package/OmniVoice-PT-Q4-Linux/modelos/
COPY packaging/linux-q4/gerar.sh packaging/linux-q4/LEIA-ME.txt /package/OmniVoice-PT-Q4-Linux/
RUN chmod 0755 /package/OmniVoice-PT-Q4-Linux/omnivoice-pt \
               /package/OmniVoice-PT-Q4-Linux/gerar.sh \
 && chmod 0644 /package/OmniVoice-PT-Q4-Linux/LEIA-ME.txt \
               /package/OmniVoice-PT-Q4-Linux/modelos/*

FROM scratch AS export
COPY --from=package /package/ /

FROM alpine:3.20 AS zip
RUN apk add --no-cache zip
COPY --from=package /package/ /package/
WORKDIR /package
RUN zip -9 -r /OmniVoice-PT-Q4-Linux.zip OmniVoice-PT-Q4-Linux

FROM scratch AS zip-export
COPY --from=zip /OmniVoice-PT-Q4-Linux.zip /
