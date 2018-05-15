#include "cuda_graph.h"
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_X 128
#if BLOCK_DIM_Y * BLOCK_DIM_X > 1024  // max threads per block
#error "Block size too big"
#endif

__device__ bool edge_exist(node_t a, node_t b, cuda_edge_t* edges,
                           size_t edge_count) {
  for (int ei = 0; ei < edge_count; ei++) {
    cuda_edge_t edge = edges[ei];
    if (edge.a == a && edge.b == b) {
      return true;
    }
  }
  return false;
}

__global__ void cuda_generate_k_triangles(node_t* nodes, size_t node_count,
                                          cuda_edge_t* edges, size_t edge_count,
                                          cuda_triangle_t* triangles,
                                          size_t* triangle_count) {
  size_t edge_id = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
  size_t node_id = threadIdx.y + blockIdx.y * BLOCK_DIM_Y;
  if (edge_id < edge_count && node_id < node_count) {
    node_t node = nodes[node_id];
    cuda_edge_t edge = edges[edge_id];
    if (node > edge.a) {
      return;
    }

    if (edge_exist(node, edge.a, edges, edge_count) &&
        edge_exist(node, edge.b, edges, edge_count)) {
      size_t ti = atomicAdd((unsigned long long*)triangle_count, 1);
      cuda_triangle_t triangle;
      triangle.a = node;
      triangle.b = edge.a;
      triangle.c = edge.b;
      triangles[ti] = triangle;
    }
  }
}
