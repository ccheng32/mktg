DEBUG=n
OMP=y
WARN=y
BASEFLAGS= --std=c++11
CPPFLAGS= $(BASEFLAGS)
NVCCFLAGS= $(BASEFLAGS)
CC=g++

CUDALIB=/usr/local/cuda-8.0
NVCC=$(CUDALIB)/bin/nvcc
CUDACC=sm_61
LIBS=-L$(CUDALIB)/lib64 -lcudart

ifeq ($(DEBUG),n)
	CPPFLAGS+= -O3
	NVCCFLAGS += -O3
else
	DEBUGFLAG= -O0 -g -DDEBUG
	CPPFLAGS+= $(DEBUGFLAG)
	NVCCFLAGS+= $(DEBUGFLAG) -G
endif

ifeq ($(WARN),y)
	CPPFLAGS+= -Wall
	NVCCFLAGS += --compiler-options -Wall
endif

ifeq ($(OMP),y)
	OMPFLAG=-fopenmp
	CPPFLAGS+= $(OMPFLAG)
	NVCCFLAGS+= -Xcompiler $(OMPFLAG)
endif

OBJ=main.o graph.o tera.o cuda_generate_k_triangles.o

mktg: $(OBJ)
	$(CC) $(CPPFLAGS) $(OBJ) $(LIBS) -o mktg

main.o: main.cpp graph.h
	$(CC) $(CPPFLAGS) -c main.cpp

tera.o: tera.cpp graph.h
	$(CC) $(CPPFLAGS) -c tera.cpp

graph.o: graph.cpp graph.h cuda_graph.h
	$(CC) $(CPPFLAGS) -c graph.cpp

cuda_generate_k_triangles.o: cuda_generate_k_triangles.cu cuda_graph.h graph.h
	$(NVCC) $(NVCCFLAGS) -c --gpu-architecture=$(CUDACC) cuda_generate_k_triangles.cu


clean:
	rm -f *.o mktg
