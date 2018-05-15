#include "cuda_graph.h"
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_X 128
#if BLOCK_DIM_Y * BLOCK_DIM_X > 1024  // max threads per block
#error "Block size too big"
#endif

#define NODE_TRIANGLE_RATIO 100000

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
__device__ bool __edge_exist(node_t a, node_t b, const cuda_edge_t* edges,
                             size_t edge_count) {
  for (int ei = 0; ei < edge_count; ei++) {
    cuda_edge_t edge = edges[ei];
    if (edge.a == a && edge.b == b) {
      return true;
    }
  }
  return false;
}

__global__ void _cuda_generate_k_triangles(
    const node_t* nodes, size_t node_count, const cuda_edge_t* edges,
    size_t edge_count, cuda_triangle_t* triangles, size_t* triangle_count) {
  size_t edge_id = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
  size_t node_id = threadIdx.y + blockIdx.y * BLOCK_DIM_Y;
  if (edge_id < edge_count && node_id < node_count) {
    node_t node = nodes[node_id];
    cuda_edge_t edge = edges[edge_id];
    if (node < edge.a) {
      return;
    }

    if (__edge_exist(node, edge.a, edges, edge_count) &&
        __edge_exist(node, edge.b, edges, edge_count)) {
      size_t ti = atomicAdd((unsigned long long*)triangle_count, 1);
      printf("gpu triangle %lu: %u %u %u\n", ti + 1, node, edge.a, edge.b);
      cuda_triangle_t triangle;
      triangle.a = node;
      triangle.b = edge.a;
      triangle.c = edge.b;
      triangles[ti] = triangle;
    }
  }
}

void cuda_generate_k_triangles(
    const std::vector<node_t>& nodes,
    const std::vector<std::pair<node_t, node_t>>& edges,
    std::vector<std::tuple<node_t, node_t, node_t>>& triangle_k) {
  // copy nodes
  node_t* d_nodes;
  gpuErrchk(cudaMalloc(&d_nodes, sizeof(node_t) * nodes.size()));
  gpuErrchk(cudaMemcpy(d_nodes, nodes.data(), sizeof(node_t) * nodes.size(),
                       cudaMemcpyHostToDevice));

  // copy edges
  cuda_edge_t* d_edges;
  gpuErrchk(cudaMalloc(&d_edges, sizeof(cuda_edge_t) * edges.size()));
  for (size_t ei = 0; ei < edges.size(); ei++) {
    cuda_edge_t edge;
    edge.a = edges[ei].first;
    edge.b = edges[ei].second;
    gpuErrchk(cudaMemcpy(d_edges + ei, &edge, sizeof(cuda_edge_t),
                         cudaMemcpyHostToDevice));
  }

  // allocate memory for triangles
  cuda_triangle_t* d_triangles;
  gpuErrchk(
      cudaMalloc(&d_triangles, sizeof(cuda_triangle_t) * NODE_TRIANGLE_RATIO));

  size_t dim_nodes = (nodes.size() - 1) / BLOCK_DIM_Y + 1;
  size_t dim_edges = (edges.size() - 1) / BLOCK_DIM_X + 1;
  dim3 dim_grid(dim_edges, dim_nodes);
  dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y);

  size_t num_triangles = 0;
  size_t* d_num_triangles;
  gpuErrchk(cudaMalloc(&d_num_triangles, sizeof(size_t)));

  _cuda_generate_k_triangles<<<dim_grid, dim_block>>>(
      d_nodes, nodes.size(), d_edges, edges.size(), d_triangles,
      d_num_triangles);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(&num_triangles, d_num_triangles, sizeof(node_t),
                       cudaMemcpyDeviceToHost));

  printf("gpu %lu triangles\n", num_triangles);

  gpuErrchk(cudaFree(d_nodes));
  gpuErrchk(cudaFree(d_edges));
  gpuErrchk(cudaFree(d_triangles));
}
