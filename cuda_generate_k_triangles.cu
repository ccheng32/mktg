#include <algorithm>
#include <limits>
#include "cuda_graph.h"
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_X 128
#if BLOCK_DIM_Y * BLOCK_DIM_X > 1024  // max threads per block
#error "Block size too big"
#endif

#define NODE_TRIANGLE_RATIO 1000
#define EDGES_PER_THREAD 1

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
                             size_t node_start, size_t node_degree) {
  int l = node_start;
  int r = node_start + node_degree;
  while (l <= r) {
    int m = l + (r - l) / 2;
    auto edge = edges[m];

    // Check if x is present at mid
    if (edge.b == b) return true;

    // If x greater, ignore left half
    if (edge.b < b) l = m + 1;

    // If x is smaller, ignore right half
    else
      r = m - 1;
  }

  return false;
}

__global__ void _cuda_generate_k_triangles(
    const node_t* nodes, const size_t* node_starts, size_t node_count,
    const cuda_edge_t* edges, size_t edge_count, cuda_triangle_t* triangles,
    size_t* triangle_count) {
  size_t first_edge_id =
      (threadIdx.x + blockIdx.x * BLOCK_DIM_X) * EDGES_PER_THREAD;
  size_t node_id = threadIdx.y + blockIdx.y * BLOCK_DIM_Y;
  for (size_t edge_id = first_edge_id;
       edge_id < first_edge_id + EDGES_PER_THREAD; edge_id++) {
    if (edge_id < edge_count && node_id < node_count) {
      node_t node = nodes[node_id];
      size_t node_start = node_starts[node_id];
      size_t node_degree = node_starts[node_id + 1] - node_start;
      cuda_edge_t edge = edges[edge_id];
      if (node <= edge.a) {
        continue;
      }

      if (__edge_exist(node, edge.a, edges, node_start, node_degree) &&
          __edge_exist(node, edge.b, edges, node_start, node_degree)) {
        size_t ti = atomicAdd((unsigned long long*)triangle_count, 1);
#ifdef DEBUG
        printf("gpu triangle %lu: %u %u %u\n", ti + 1, node, edge.a, edge.b);
#endif
        // cuda_triangle_t triangle;
        // triangle.a = node;
        // triangle.b = edge.a;
        // triangle.c = edge.b;
        //    triangles[ti] = triangle;
      }
    }
  }
}

void cuda_generate_k_triangles(
    std::vector<node_t> nodes, std::vector<std::pair<node_t, node_t>> edges,
    std::vector<std::tuple<node_t, node_t, node_t>>& triangle_k) {
  // sort the edges first
  std::sort(edges.begin(), edges.end());
  std::sort(nodes.begin(), nodes.end());

  // preprocess edge starts
  std::vector<size_t> node_starts;
  std::vector<node_t> active_first_nodes;
  node_t prev_node = std::numeric_limits<node_t>::max();
  size_t node_idx = 0;
  for (size_t ei = 0; ei < edges.size(); ei++) {
    auto edge = edges[ei];
    // found a new start node
    if (edge.first != prev_node) {
      prev_node = edge.first;
      while (nodes[node_idx] != prev_node) {
        node_idx++;
      }
      active_first_nodes.push_back(nodes[node_idx]);
      node_starts.push_back(ei);
    }
  }
  node_starts.push_back(edges.size());

  // declare device pointers and allocate
  node_t* d_nodes;
  gpuErrchk(cudaMalloc(&d_nodes, sizeof(node_t) * active_first_nodes.size()));
  size_t* d_node_starts;
  gpuErrchk(cudaMalloc(&d_node_starts, sizeof(size_t) * node_starts.size()));
  cuda_edge_t* d_edges;
  gpuErrchk(cudaMalloc(&d_edges, sizeof(cuda_edge_t) * edges.size()));
  cuda_triangle_t* d_triangles;
  gpuErrchk(
      cudaMalloc(&d_triangles, sizeof(cuda_triangle_t) * NODE_TRIANGLE_RATIO));

  // copy nodes
  gpuErrchk(cudaMemcpyAsync(d_nodes, active_first_nodes.data(),
                            sizeof(node_t) * active_first_nodes.size(),
                            cudaMemcpyHostToDevice));

  // copy edge starts
  gpuErrchk(cudaMemcpyAsync(d_node_starts, node_starts.data(),
                            sizeof(size_t) * node_starts.size(),
                            cudaMemcpyHostToDevice));

  // copy edges
  for (size_t ei = 0; ei < edges.size(); ei++) {
    cuda_edge_t edge;
    edge.a = edges[ei].first;
    edge.b = edges[ei].second;
    gpuErrchk(cudaMemcpyAsync(d_edges + ei, &edge, sizeof(cuda_edge_t),
                              cudaMemcpyHostToDevice));
  }

  size_t* num_triangles;
  gpuErrchk(cudaMallocManaged(&num_triangles, sizeof(size_t)));
  *num_triangles = 0;

  size_t dim_nodes = (active_first_nodes.size() - 1) / BLOCK_DIM_Y + 1;
  size_t dim_edges = (edges.size() - 1) / (BLOCK_DIM_X * EDGES_PER_THREAD) + 1;
  dim3 dim_grid(dim_edges, dim_nodes);
  dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y);

  _cuda_generate_k_triangles<<<dim_grid, dim_block>>>(
      d_nodes, d_node_starts, active_first_nodes.size(), d_edges, edges.size(),
      d_triangles, num_triangles);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaDeviceSynchronize());

  printf("gpu %lu triangles\n", *num_triangles);

  gpuErrchk(cudaFree(d_nodes));
  gpuErrchk(cudaFree(d_node_starts));
  gpuErrchk(cudaFree(d_edges));
  gpuErrchk(cudaFree(d_triangles));
}
