NVCC ?= nvcc
ARCH ?= sm_75
TARGET := ncu_diagnostic_tree

NVCCFLAGS := -O3 -lineinfo --std=c++17 -arch=$(ARCH) -Xptxas=-v

.PHONY: all clean list

all: $(TARGET)

$(TARGET): ncu_diagnostic_tree.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

list: $(TARGET)
	./$(TARGET) --list

clean:
	rm -f $(TARGET) *.ncu-rep
