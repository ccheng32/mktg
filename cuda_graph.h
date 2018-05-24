#ifndef CUDA_GRAPH
#define CUDA_GRAPH

#define BLOCK_DIM_Y 8
#define BLOCK_DIM_X 128
#include "graph.h"

typedef size_t cuda_node_t;

typedef struct {
  node_t a;
  node_t b;
} cuda_edge_t;

typedef struct {
  node_t a;
  node_t b;
  node_t c;
} cuda_triangle_t;

void cuda_generate_k_triangles(
    std::vector<node_t> nodes, std::vector<std::pair<node_t, node_t>> edges,
    std::vector<std::tuple<node_t, node_t, node_t>>& triangle_k, size_t n);
#endif
