#include <sys/time.h>
#include "cuda_graph.h"
#include "graph.h"

std::vector<node_t> graph::tera(size_t k, size_t n) {
  this->graph_k(k);
  generate_k_triangles();

  std::vector<node_t> ans;
  get_nodes(ans);

  make_adj_list_copy();
  while (ans.size() > n) {
    struct timeval start;
    gettimeofday(&start, NULL);

    size_t max_triangle_number = 0;
    node_t max_triangle_node = adj_list.cbegin()->first;

    for (auto node_it = ans.cbegin(); node_it < ans.cend(); node_it++) {
      auto node = *node_it;
      size_t node_triangle_number = triangle_number(node);
#ifdef DEBUG
      printf("node %u has tnum %lu\n", node, node_triangle_number);
#endif
      if (node_triangle_number > max_triangle_number) {
        max_triangle_node = node;
        max_triangle_number = node_triangle_number;
      } else if (node_triangle_number == max_triangle_number) {
        if (get_degree(node) > get_degree(max_triangle_node)) {
          max_triangle_node = node;
          max_triangle_number = node_triangle_number;
        }
      }
    }

    struct timeval remove_start;
    gettimeofday(&remove_start, NULL);
    remove_node(max_triangle_node);
    get_nodes(ans);

    struct timeval end;
    gettimeofday(&end, NULL);
    //#ifdef DEBUG
    printf("removing node %u with tnum %lu, time: %lf, removal time: %lf\n",
           max_triangle_node, max_triangle_number,
           (1000000.0 * (end.tv_sec - start.tv_sec) + end.tv_usec -
            start.tv_usec) /
               1000000.0,
           (1000000.0 * (end.tv_sec - remove_start.tv_sec) + end.tv_usec -
            remove_start.tv_usec) /
               1000000.0);
    //#endif
  }

  restore_adj_list();
  this->graph_k(1);
  return ans;
}

std::vector<node_t> graph::cuda_tera(size_t k, size_t n) {
  this->graph_k(k);

  // get a copy of the k hop graph in edge list form
  std::vector<std::pair<node_t, node_t>> edge_list_k;
  get_edge_list_k(edge_list_k);

  std::vector<node_t> nodes;
  get_nodes(nodes);

  // GPU TESTING START
  struct timeval start, end;
  gettimeofday(&start, NULL);
  std::vector<std::tuple<node_t, node_t, node_t>> k_triangles_gpu;
  cuda_generate_k_triangles(nodes, edge_list_k, k_triangles_gpu, n);
  gettimeofday(&end, NULL);
  printf("in %lf seconds\n", (1000000.0 * (end.tv_sec - start.tv_sec) +
                              end.tv_usec - start.tv_usec) /
                                 1000000.0);
  // GPU TESTING END
  return std::vector<node_t>();
}
