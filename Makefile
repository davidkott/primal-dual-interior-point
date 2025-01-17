#This is a simple standalone example. See README.txt
# Initially it is setup to use OpenBLAS.
# See magma/make.inc for alternate BLAS and LAPACK libraries,
# or use pkg-config as described below.

# Paths where MAGMA, CUDA, and OpenBLAS are installed.
# MAGMADIR can be .. to test without installing.
#MAGMADIR     ?= ..
MAGMADIR     ?= /data/accounts/kott/MAGMA
CUDADIR      ?= /usr/local/cuda
OPENBLASDIR  ?= /data/accounts/kott/openBLAS
-include make.inc

CXX           = g++
CFLAGS        = -Wall
LDFLAGS       = -Wall #-fopenmp
#CXXFLAGS      = -O3 -std=c++11  --gpu-architecture=sm_80 #hail mary
CXXFLAGS      = -O3 -fopenmp #hail mary

# ----------------------------------------
# Flags and paths to MAGMA, CUDA, and LAPACK/BLAS
MAGMA_CFLAGS     := -DADD_ \
                    -I$(MAGMADIR)/include \
                    -I$(MAGMADIR)/sparse/include \
                    -I$(CUDADIR)/include \
                    -I$(OPENBLASDIR)/include

MAGMA_F90FLAGS   := -Dmagma_devptr_t="integer(kind=8)" \
                    -I$(MAGMADIR)/include

# may be lib instead of lib64 on some systems
MAGMA_LIBS       := -L$(MAGMADIR)/lib -lmagma_sparse -lmagma \
                    -L$(CUDADIR)/lib64 -lcublas -lcudart -lcusparse \
                    -L$(OPENBLASDIR)/lib -lopenblas


# ----------------------------------------
# Alternatively, using pkg-config (see README.txt):
# MAGMA_CFLAGS   := $(shell pkg-config --cflags magma)
#
# MAGMA_F90FLAGS := -Dmagma_devptr_t="integer(kind=8)" \
#                   $(shell pkg-config --cflags-only-I magma)
#
# MAGMA_LIBS     := $(shell pkg-config --libs   magma)


# ----------------------------------------
default:
	@echo "Available make targets are:"
	@echo "  make all       # compiles example_v1, example_v2, example_sparse, example_sparse_operator, and example_f"
	@echo "  make c         # compiles example_v1, example_v2, example_sparse, example_sparse_operator"
	@echo "  make cpp   # compiles you're c++ file, add script to Makefile if not seen"
	@echo "  make clean     # deletes executables and object files"

all: cpp

cpp: test single_gpu multi_gpu cpu_code

clean:
	-rm -f cpu_code test single_gpu multi_gpu *.o *.mod


#Ya boy David Kotts c++ addition

test: test.cpp
	$(CXX) $(CXXFLAGS) $(MAGMA_CFLAGS) -o test test.cpp  $(MAGMA_LIBS) -lstdc++

single_gpu: single_gpu.cpp
	$(CXX) $(CXXFLAGS) $(MAGMA_CFLAGS) -o single_gpu single_gpu.cpp  $(MAGMA_LIBS) -lstdc++

multi_gpu: multi_gpu.cpp
	$(CXX) $(CXXFLAGS) $(MAGMA_CFLAGS) -o multi_gpu multi_gpu.cpp  $(MAGMA_LIBS) -lstdc++
 
cpu_code: cpu_code.cpp
	$(CXX) $(CXXFLAGS) $(MAGMA_CFLAGS) -o cpu_code cpu_code.cpp  $(MAGMA_LIBS) -lstdc++