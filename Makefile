BUILD_DIR  := build
SRC_DIR    := src
INCLUDE    := include
DATASET_DIR:= data/DIV2K_valid_LR_bicubic/X4
CIMG_H     := $(INCLUDE)/CImg.h

GXX        := g++-9
NVCC       := nvcc
ARCH       := sm_86
CUDA_DIR   := /usr/lib/cuda
CXXFLAGS   := -std=c++17 -O3 -I$(INCLUDE) -I$(SRC_DIR) -I$(CUDA_DIR)/include
NVFLAGS    := -ccbin $(GXX) -arch=$(ARCH) -O3 -use_fast_math -std=c++17
INCFLAGS   := -I$(INCLUDE) -I$(SRC_DIR)
LDFLAGS    := -lpng -lz -lcudart

TARGET     := $(BUILD_DIR)/covariance

.PHONY: all setup data clean run1 run2 run

all: setup $(TARGET)

setup: $(CIMG_H)

$(CIMG_H):
	bash scripts/download_cimg.sh

data:
	bash scripts/download_dataset.sh

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/preprocess.o: $(SRC_DIR)/preprocess.cpp $(SRC_DIR)/dataset.h $(CIMG_H) | $(BUILD_DIR)
	$(GXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu $(SRC_DIR)/utils.cuh $(SRC_DIR)/dataset.h \
                      $(SRC_DIR)/kernels.cuh $(SRC_DIR)/experiment1.cuh $(SRC_DIR)/experiment2.cuh | $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) $(INCFLAGS) -c $< -o $@

$(TARGET): $(BUILD_DIR)/main.o $(BUILD_DIR)/preprocess.o
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDFLAGS)

run1: $(TARGET) data
	$(TARGET) --dir $(DATASET_DIR) --size 32 --run 1

run2: $(TARGET) data
	$(TARGET) --dir $(DATASET_DIR) --size 32 --run 2

run: $(TARGET) data
	$(TARGET) --dir $(DATASET_DIR) --size 32 --run both

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -f $(CIMG_H)
	rm -rf data/DIV2K_valid_LR_bicubic data/DIV2K_valid_LR_bicubic_X4.zip
