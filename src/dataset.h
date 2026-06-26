#pragma once

struct Dataset {
    float* h_data = nullptr;
    int    m      = 0;
    int    n      = 0;
    int    width  = 0;
    int    height = 0;
};

Dataset load_dataset(const char* dir, int target_size);
void    free_dataset(Dataset& ds);
