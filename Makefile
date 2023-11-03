EXEC = string-art
SRC = $(notdir $(wildcard src/*.cc)) 
CUDA_SRC = $(notdir $(wildcard src/*.cu))
OBJECTS = $(addprefix build/,$(SRC:.cc=.o)) 
CUDA_OBJECTS = $(addprefix build/,$(CUDA_SRC:.cu=.o))
ALL_OBJECTS = $(OBJECTS) $(CUDA_OBJECTS)
DEPENDS = $(OBJECTS:.o=.d)

CXX = g++
NVCC = /usr/local/cuda-12.2/bin/nvcc -std=c++20 -ccbin g++

CXXFLAGS = -Ofast -std=c++20 -Wall -Werror=vla -MMD
DBG_CXXFLAGS = -g -std=c++20 -Wall -Werror=vla -MMD
# CUDA doesn't support -MMD since it makes some .cpp files in /tmp/
# See https://icl.utk.edu/~mgates3/gpu-tutorial/cuda-examples/Makefile
NVCC_CXXFLAGS := -Xcompiler '$(filter-out -MMD, $(CXXFLAGS))' #-arch=sm_89
NVCC_DBG_CXXFLAGS := -Xcompiler '$(filter-out -MMD, $(DBG_CXXFLAGS))'

#LIBS = $(shell Magick++-config --cppflags --cxxflags)
#LDFLAGS = $(shell Magick++-config --ldflags --libs) -fopenmp
LIBS = -I/usr/include/ImageMagick-6 -fopenmp -DMAGICKCORE_HDRI_ENABLE=1 -DMAGICKCORE_QUANTUM_DEPTH=16 
LDFLAGS = -lMagick++-6.Q16HDRI -lMagickWand-6.Q16HDRI -lMagickCore-6.Q16HDRI -fopenmp -L/usr/lib/wsl/lib -L/usr/local/cuda-12.2/lib64/ -lcuda -lcudart
NVCC_LIBS := -Xcompiler '$(LIBS)'
NVCC_LDFLAGS := -Xcompiler '$(LDFLAGS)' 
#-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment

# See https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/
# for information about separate compilation with CUDA
$(EXEC): $(ALL_OBJECTS)
	$(NVCC) -dlink $(CUDA_OBJECTS) $(NVCC_LDFLAGS) -o build/cuda-$(EXEC).o
	$(CXX) $(ALL_OBJECTS) build/cuda-$(EXEC).o $(LDFLAGS) -o $(EXEC)

build/%.o: src/%.cc
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(LIBS) -c $< -o $@

build/%.o: src/%.cu
	@mkdir -p build
	# See https://stackoverflow.com/questions/31006581/cuda-device-unresolved-extern-function
	# for info about the -dc option
	$(NVCC) $(NVCC_CXXFLAGS) $(NVCC_LIBS) -dc -c $< -o $@

.PHONY: debug

debug: $(ALL_OBJECTS)
	$(CXX) $(DBG_CXXFLAGS) $(LIBS) $(LDFLAGS) $(addprefix src/, $(SRC)) -o $(EXEC)

-include $(DEPENDS)

.PHONY: clean

clean:
	rm $(ALL_OBJECTS) $(EXEC) $(DEPENDS) build/cuda-$(EXEC).o
