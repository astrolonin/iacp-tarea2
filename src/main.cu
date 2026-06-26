#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "utils.cuh"
#include "dataset.h"
#include "experiment1.cuh"
#include "experiment2.cuh"

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  --dir PATH       Dataset directory [default: data/DIV2K_valid_LR_bicubic_X4]\n"
        "  --size N         Resize images to NxN grayscale [default: 128]\n"
        "  --streams LIST   Comma-separated stream counts for Exp2 [default: 1,2,4,8,16]\n"
        "  --run MODE       1=Exp1, 2=Exp2, both [default: both]\n"
        "  --help           Show this message\n",
        prog);
}

static std::vector<int> parse_streams(const char* arg) {
    std::vector<int> v;
    const char* p = arg;
    while (*p) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

int main(int argc, char** argv) {
    const char* dir     = "data/DIV2K_valid_LR_bicubic/X4";
    int         size    = 32;
    const char* mode    = "both";
    const char* streams_arg = "1,2,4,8,16";

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--dir")     && i + 1 < argc) dir     = argv[++i];
        else if (!std::strcmp(argv[i], "--size")    && i + 1 < argc) size    = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--streams") && i + 1 < argc) streams_arg = argv[++i];
        else if (!std::strcmp(argv[i], "--run")     && i + 1 < argc) mode    = argv[++i];
        else if (!std::strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { usage(argv[0]); return 1; }
    }

    std::printf("=== Tarea 2 — Covariance Matrix on CUDA ===\n");
    std::printf("Dataset: %s | Resize: %dx%d | n=%d\n", dir, size, size, size * size);

    Dataset ds = load_dataset(dir, size);
    int n = ds.n;
    int m = ds.m;

    if (!std::strcmp(mode, "1") || !std::strcmp(mode, "both")) {
        std::printf("\n--- Experiment 1: Traditional CUDA (Stream 0) ---\n");
        Exp1Result r = run_experiment1(ds.h_data, m, n);
        std::printf("RESULT,exp1,h2d,%.3f\n", r.ms_h2d);
        std::printf("RESULT,exp1,kernels,%.3f\n", r.ms_kernels);
        std::printf("RESULT,exp1,d2h,%.3f\n", r.ms_d2h);
        std::printf("RESULT,exp1,total,%.3f\n", r.ms_total);
    }

    if (!std::strcmp(mode, "2") || !std::strcmp(mode, "both")) {
        auto streams_list = parse_streams(streams_arg);
        std::printf("\n--- Experiment 2: CUDA Streams ---\n");
        for (int S : streams_list) {
            std::printf("\n");
            Exp2Result r = run_experiment2(ds.h_data, m, n, S);
            std::printf("RESULT,exp2,S%d,total,%.3f\n", S, r.ms_total);
            std::printf("RESULT,exp2,S%d,batches,%d\n", S, r.num_batches);
            std::printf("RESULT,exp2,S%d,batch_size,%d\n", S, r.batch_size);
        }
    }

    free_dataset(ds);
    std::printf("\nDone.\n");
    return 0;
}
