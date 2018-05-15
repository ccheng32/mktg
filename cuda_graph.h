#ifndef CUDA_GRAPH
#define CUDA_GRAPH

#include "graph.h"

typedef struct {
  node_t a;
  node_t b;
} cuda_edge_t;

typedef struct {
  node_t a;
  node_t b;
  node_t c;
} cuda_triangle_t;

#endif
