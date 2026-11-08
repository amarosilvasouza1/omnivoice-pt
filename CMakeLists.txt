cmake_minimum_required(VERSION 3.20)
project(omnivoice-pt C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(OVPT_BUILD_TESTS "Build OmniVoice-PT validation executables" ON)
option(OVPT_FULLY_STATIC "Link a self-contained Linux executable" OFF)
option(OVPT_OPENMP "Parallelize the sampler with OpenMP" ON)

if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
endif()

# exes junto com os DLLs do ggml
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# GGML (backend CPU sempre; CUDA ou Vulkan opcionais)
add_subdirectory(third_party/ggml)

# --- biblioteca do runtime ---
add_library(ovpt STATIC
    src/ov_tokenizer.c
    src/ov_model.c
    src/ov_sampler.c
    src/ov_decoder.c
)
target_include_directories(ovpt PUBLIC src)
target_link_libraries(ovpt PUBLIC ggml)
if(GGML_CUDA)
    target_compile_definitions(ovpt PRIVATE GGML_USE_CUDA)
endif()
if(GGML_VULKAN)
    target_compile_definitions(ovpt PRIVATE GGML_USE_VULKAN)
endif()

# O laço de CFG/log_softmax do sampler é ~30% do tempo de geração e é
# perfeitamente paralelo por posição (a saída continua bit-idêntica: o gumbel
# de posição é sorteado em série antes do laço). O binário estático do Linux
# fica de fora — ele é compilado sem OpenMP de propósito, para a CPU antiga.
if(OVPT_OPENMP AND NOT OVPT_FULLY_STATIC)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(ovpt PUBLIC OpenMP::OpenMP_C)
        message(STATUS "OmniVoice-PT: sampler com OpenMP")
    else()
        message(STATUS "OmniVoice-PT: OpenMP nao encontrado, sampler serial")
    endif()
endif()

# --- executável principal ---
add_executable(omnivoice-pt src/main.c)
target_link_libraries(omnivoice-pt PRIVATE ovpt)

if(UNIX AND NOT APPLE)
    target_compile_definitions(ovpt PRIVATE _FILE_OFFSET_BITS=64)
    target_compile_definitions(omnivoice-pt PRIVATE _FILE_OFFSET_BITS=64)
endif()

if(OVPT_FULLY_STATIC AND UNIX AND NOT APPLE)
    target_link_options(omnivoice-pt PRIVATE -static)
endif()

install(TARGETS omnivoice-pt RUNTIME DESTINATION bin)

# --- testes ---
if(OVPT_BUILD_TESTS)
enable_testing()

add_executable(test_tokenizer tests/test_tokenizer.c)
target_link_libraries(test_tokenizer PRIVATE ovpt)
add_test(NAME tokenizer COMMAND test_tokenizer
    ${CMAKE_SOURCE_DIR}/build/tokenizer.bin
    ${CMAKE_SOURCE_DIR}/build/tokenizer_golden.bin)

add_executable(test_model tests/test_model.c)
target_link_libraries(test_model PRIVATE ovpt)
add_test(NAME model COMMAND test_model
    ${CMAKE_SOURCE_DIR}/build/omnivoice-pt-f16.gguf
    ${CMAKE_SOURCE_DIR}/build/model_golden.bin)

add_executable(test_decoder tests/test_decoder.c)
target_link_libraries(test_decoder PRIVATE ovpt)
add_test(NAME decoder COMMAND test_decoder
    ${CMAKE_SOURCE_DIR}/build/higgs-decoder-f16.gguf
    ${CMAKE_SOURCE_DIR}/build/decoder_golden.bin)
endif()
