#include <algorithm>
#include <limits>
#include "cuda_graph.h"
#if BLOCK_DIM_Y * BLOCK_DIM_X > 1024  // max threads per block
#error "Block size too big"
#endif

#define MAX_TRIANGLE_NUM 2000000000
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

__device__ bool __cuda_edge_exist(cuda_node_t a, cuda_node_t b,
                                  const cuda_edge_t* edges, size_t node_start,
                                  size_t node_degree) {
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
    const cuda_node_t* nodes, const size_t* node_starts, size_t node_count,
    const cuda_edge_t* edges, size_t edge_count, cuda_triangle_t* triangles,
    size_t* triangle_count, size_t* triangles_per_node) {
  size_t first_edge_id =
      (threadIdx.x + blockIdx.x * BLOCK_DIM_X) * EDGES_PER_THREAD;
  size_t node_id = threadIdx.y + blockIdx.y * BLOCK_DIM_Y;
  for (size_t edge_id = first_edge_id;
       edge_id < first_edge_id + EDGES_PER_THREAD; edge_id++) {
    if (edge_id < edge_count && node_id < node_count) {
      cuda_node_t node = nodes[node_id];
      size_t node_start = node_starts[node_id];
      size_t node_degree = node_starts[node_id + 1] - node_start;
      cuda_edge_t edge = edges[edge_id];
      if (node <= edge.a) {
        continue;
      }

      if (__cuda_edge_exist(node, edge.a, edges, node_start, node_degree) &&
          __cuda_edge_exist(node, edge.b, edges, node_start, node_degree)) {
        size_t ti = atomicAdd((unsigned long long*)triangle_count, 1);
        atomicAdd((unsigned long long*)&(triangles_per_node[node]), 1);
        atomicAdd((unsigned long long*)&(triangles_per_node[edge.a]), 1);
        atomicAdd((unsigned long long*)&(triangles_per_node[edge.b]), 1);
#ifdef DEBUG
        printf("gpu triangle %lu: %u %u %u\n", ti + 1, node, edge.a, edge.b);
#endif
        cuda_triangle_t triangle;
        triangle.a = node;
        triangle.b = edge.a;
        triangle.c = edge.b;
        triangles[ti] = triangle;
      }
    }
  }
}
__global__ void _cuda_remove_max_node(const cuda_triangle_t* triangles,
                                      size_t* triangles_per_node,
                                      bool* triangle_is_active,
                                      cuda_node_t max_node,
                                      size_t triangle_count) {
  size_t tri_idx = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
  if (tri_idx < triangle_count) {
    if (!triangle_is_active[tri_idx]) {
      return;
    }
    cuda_triangle_t triangle = triangles[tri_idx];
    if (triangle.a == max_node || triangle.b == max_node ||
        triangle.c == max_node) {
      triangle_is_active[tri_idx] = false;
      atomicAdd((int*)&(triangles_per_node[triangle.a]), -1);
      atomicAdd((int*)&(triangles_per_node[triangle.b]), -1);
      atomicAdd((int*)&(triangles_per_node[triangle.c]), -1);
    }
  }
}

void build_cuda_node_map(std::unordered_map<node_t, cuda_node_t>& graph_to_cuda,
                         std::unordered_map<cuda_node_t, node_t>& cuda_to_graph,
                         std::vector<node_t>& nodes) {
  graph_to_cuda.clear();
  cuda_to_graph.clear();
  for (size_t ii = 0; ii < nodes.size(); ii++) {
    graph_to_cuda[nodes[ii]] = ii;
    cuda_to_graph[ii] = nodes[ii];
  }
}

void cuda_generate_k_triangles(
    std::vector<node_t> nodes, std::vector<std::pair<node_t, node_t>> edges,
    std::vector<std::tuple<node_t, node_t, node_t>>& triangle_k, size_t n) {
  // build a map for mapping cuda index and original graph nodes
  std::unordered_map<node_t, cuda_node_t> graph_to_cuda;
  std::unordered_map<cuda_node_t, node_t> cuda_to_graph;
  build_cuda_node_map(graph_to_cuda, cuda_to_graph, nodes);

  std::vector<cuda_node_t> cuda_nodes;
  std::vector<std::pair<cuda_node_t, cuda_node_t>> cuda_edges;

  // map nodes and edges to the new indices
  for (auto& node : nodes) {
    cuda_nodes.push_back(graph_to_cuda.find(node)->second);
  }
  nodes.clear();
  for (auto& edge : edges) {
    cuda_node_t a = graph_to_cuda.find(edge.first)->second;
    cuda_node_t b = graph_to_cuda.find(edge.second)->second;
    if (a < b) {
      std::swap(a, b);
    }
    cuda_edges.push_back(std::make_pair(a, b));
  }
  edges.clear();

  // sort the edges and nodes first
  std::sort(cuda_edges.begin(), cuda_edges.end());
  std::sort(cuda_nodes.begin(), cuda_nodes.end());

  // preprocess edge starts
  std::vector<size_t> node_starts;
  std::vector<cuda_node_t> active_first_nodes;
  cuda_node_t prev_node = std::numeric_limits<cuda_node_t>::max();
  size_t node_idx = 0;
  for (size_t ei = 0; ei < cuda_edges.size(); ei++) {
    auto edge = cuda_edges[ei];
    // found a new start node
    if (edge.first != prev_node) {
      prev_node = edge.first;
      while (cuda_nodes[node_idx] != prev_node) {
        node_idx++;
      }
      active_first_nodes.push_back(cuda_nodes[node_idx]);
      node_starts.push_back(ei);
    }
  }
  node_starts.push_back(cuda_edges.size());

  // declare device pointers and allocate
  cuda_node_t* d_nodes;
  gpuErrchk(
      cudaMalloc(&d_nodes, sizeof(cuda_node_t) * active_first_nodes.size()));
  size_t* d_node_starts;
  gpuErrchk(cudaMalloc(&d_node_starts, sizeof(size_t) * node_starts.size()));
  cuda_edge_t* d_edges;
  gpuErrchk(cudaMalloc(&d_edges, sizeof(cuda_edge_t) * cuda_edges.size()));
  cuda_triangle_t* d_triangles;
  gpuErrchk(cudaMallocManaged(&d_triangles,
                              sizeof(cuda_triangle_t) * MAX_TRIANGLE_NUM));
  size_t* d_triangles_per_node;
  gpuErrchk(cudaMallocManaged(&d_triangles_per_node,
                              sizeof(size_t) * cuda_nodes.size()));
  gpuErrchk(cudaMemset(d_triangles_per_node, 0x0,
                       sizeof(size_t) * cuda_nodes.size()));

  // copy nodes
  gpuErrchk(cudaMemcpyAsync(d_nodes, active_first_nodes.data(),
                            sizeof(cuda_node_t) * active_first_nodes.size(),
                            cudaMemcpyHostToDevice));

  // copy edge starts
  gpuErrchk(cudaMemcpyAsync(d_node_starts, node_starts.data(),
                            sizeof(size_t) * node_starts.size(),
                            cudaMemcpyHostToDevice));

  // copy edges
  for (size_t ei = 0; ei < cuda_edges.size(); ei++) {
    cuda_edge_t edge;
    edge.a = cuda_edges[ei].first;
    edge.b = cuda_edges[ei].second;
    gpuErrchk(cudaMemcpyAsync(d_edges + ei, &edge, sizeof(cuda_edge_t),
                              cudaMemcpyHostToDevice));
  }

  size_t* num_triangles;
  gpuErrchk(cudaMallocManaged(&num_triangles, sizeof(size_t)));
  *num_triangles = 0;

  size_t dim_nodes = (active_first_nodes.size() - 1) / BLOCK_DIM_Y + 1;
  size_t dim_edges =
      (cuda_edges.size() - 1) / (BLOCK_DIM_X * EDGES_PER_THREAD) + 1;
  dim3 dim_grid(dim_edges, dim_nodes);
  dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y);

  _cuda_generate_k_triangles<<<dim_grid, dim_block>>>(
      d_nodes, d_node_starts, active_first_nodes.size(), d_edges,
      cuda_edges.size(), d_triangles, num_triangles, d_triangles_per_node);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaDeviceSynchronize());

  printf("gpu %lu triangles\n", *num_triangles);
  gpuErrchk(cudaFree(d_nodes));
  gpuErrchk(cudaFree(d_node_starts));
  gpuErrchk(cudaFree(d_edges));

  // start tera algorithm
  bool* d_node_is_active;
  gpuErrchk(
      cudaMallocManaged(&d_node_is_active, sizeof(bool) * cuda_nodes.size()));
  gpuErrchk(
      cudaMemset(d_node_is_active, true, sizeof(bool) * cuda_nodes.size()));
  bool* d_triangle_is_active;
  gpuErrchk(
      cudaMallocManaged(&d_triangle_is_active, sizeof(bool) * *num_triangles));
  gpuErrchk(
      cudaMemset(d_triangle_is_active, true, sizeof(bool) * *num_triangles));

  for (size_t i = 0; i < cuda_nodes.size() - n; i++) {
    long long max_tri_num = -1;
    cuda_node_t max_node = 0;
    for (size_t ii = 0; ii < cuda_nodes.size(); ii++) {
      if (d_node_is_active[ii] &&
          max_tri_num < (long long)d_triangles_per_node[ii]) {
        d_node_is_active[ii] = false;
        max_tri_num = d_triangles_per_node[ii];
        max_node = cuda_nodes[ii];
      }
    }
    //#ifdef DEBUG
    printf("cuda max node: %u with trinum %lld\n",
           cuda_to_graph.find(max_node)->second, max_tri_num);
    //#endif

    // remove max node
    _cuda_remove_max_node<<<(*num_triangles - 1) / BLOCK_DIM_X + 1,
                            BLOCK_DIM_X>>>(d_triangles, d_triangles_per_node,
                                           d_triangle_is_active, max_node,
                                           *num_triangles);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  gpuErrchk(cudaFree(d_triangles));
  gpuErrchk(cudaFree(d_triangles_per_node));
  gpuErrchk(cudaFree(d_node_is_active));
  gpuErrchk(cudaFree(d_triangle_is_active));
}
