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

void cuda_generate_k_triangles(
    const std::vector<node_t>& nodes,
    const std::vector<std::pair<node_t, node_t>>& edges,
    std::vector<std::tuple<node_t, node_t, node_t>>& triangle_k);
#endif
