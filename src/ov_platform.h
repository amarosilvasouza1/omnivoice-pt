#ifndef OV_PLATFORM_H
#define OV_PLATFORM_H

#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

static inline int ov_file_seek(FILE *f, int64_t offset) {
    return _fseeki64(f, offset, SEEK_SET);
}

static inline double ov_time_seconds(void) {
    LARGE_INTEGER counter, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

static inline int ov_logical_cpu_count(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
}

#else
#include <time.h>
#include <unistd.h>

static inline int ov_file_seek(FILE *f, int64_t offset) {
    /* Linux x86_64 usa long de 64 bits; _FILE_OFFSET_BITS=64 tambem e
     * definido pelo CMake para manter isto correto em outros alvos Linux. */
    return fseek(f, (long)offset, SEEK_SET);
}

static inline double ov_time_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline int ov_logical_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 && n <= 1024 ? (int)n : 1;
}
#endif

#endif /* OV_PLATFORM_H */
