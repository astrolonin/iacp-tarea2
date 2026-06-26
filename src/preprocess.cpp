#define cimg_display 0
#define cimg_use_png
#include "CImg.h"

#include "dataset.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cstdio>

namespace fs = std::filesystem;

Dataset load_dataset(const char* dir, int target_size) {
    std::vector<std::string> png_files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png")
            png_files.push_back(entry.path().string());
    }
    std::sort(png_files.begin(), png_files.end());

    Dataset ds;
    ds.m      = (int)png_files.size();
    ds.width  = target_size;
    ds.height = target_size;
    ds.n      = target_size * target_size;

    if (ds.m == 0) {
        std::fprintf(stderr, "No PNG files found in %s\n", dir);
        std::exit(EXIT_FAILURE);
    }
    std::printf("[PRE] Found %d PNG images. Resize to %dx%d grayscale.\n",
                ds.m, target_size, target_size);

    cudaMallocHost((void**)&ds.h_data, (size_t)ds.m * ds.n * sizeof(float));

    for (int k = 0; k < ds.m; ++k) {
        cimg_library::CImg<unsigned char> img(png_files[k].c_str());
        img.resize(target_size, target_size, 1, img.spectrum());

        float* row = ds.h_data + (size_t)k * ds.n;
        if (img.spectrum() >= 3) {
            cimg_library::CImg<unsigned char> gray =
                img.get_RGBtoYCbCr().channel(0);
            gray.resize(target_size, target_size, 1, 1);
            for (int y = 0; y < target_size; ++y)
                for (int x = 0; x < target_size; ++x)
                    row[y * target_size + x] = gray(x, y) / 255.0f;
        } else {
            for (int y = 0; y < target_size; ++y)
                for (int x = 0; x < target_size; ++x)
                    row[y * target_size + x] = img(x, y) / 255.0f;
        }
    }
    std::printf("[PRE] Dataset ready: %d images, n=%d, pinned memory %.2f MB\n",
                ds.m, ds.n, (double)(ds.m * ds.n * sizeof(float)) / (1 << 20));
    return ds;
}

void free_dataset(Dataset& ds) {
    if (ds.h_data) { cudaFreeHost(ds.h_data); ds.h_data = nullptr; }
}
