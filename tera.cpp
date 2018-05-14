#include <sys/time.h>
#include "graph.h"

size_t graph::triangle_number(node_t node) {
  size_t num = 0;
  for (auto iter = adj_list_k.cbegin(); iter != adj_list_k.cend(); iter++) {
    node_t a = iter->first;
    if (!has_edge_k(a, node)) {
      continue;
    }
    for (node_t b : iter->second) {
      if (b > a && has_edge_k(node, b)) {
        num++;
      }
    }
  }

  return num;
}

std::vector<node_t> graph::tera(size_t k, size_t n) {
  this->graph_k(k);

  std::vector<node_t> ans = get_nodes_k();

  make_adj_list_copy();
  while (ans.size() > n) {
    struct timeval start;
    gettimeofday(&start, NULL);

    size_t max_triangle_number = 0;
    node_t max_triangle_node = adj_list_k.cbegin()->first;

#pragma omp parallel
    {
      size_t local_max_triangle_number = 0;
      node_t local_max_triangle_node = adj_list_k.cbegin()->first;
#pragma omp for schedule(dynamic)
      for (auto node_it = ans.cbegin(); node_it < ans.cend(); node_it++) {
        auto node = *node_it;
        size_t node_triangle_number = triangle_number(node);
#ifdef DEBUG
        printf("node %u has tnum %lu\n", node, node_triangle_number);
#endif
        if (node_triangle_number > local_max_triangle_number) {
          local_max_triangle_node = node;
          local_max_triangle_number = node_triangle_number;
        } else if (node_triangle_number == local_max_triangle_number) {
          if (get_degree(node) > get_degree(local_max_triangle_node)) {
            local_max_triangle_node = node;
            local_max_triangle_number = node_triangle_number;
          }
        }
      }

#pragma omp critical
      {
        if (local_max_triangle_number > max_triangle_number) {
          max_triangle_node = local_max_triangle_node;
          max_triangle_number = local_max_triangle_number;
        } else if (max_triangle_number == local_max_triangle_number) {
          if (get_degree(local_max_triangle_node) >
              get_degree(max_triangle_node)) {
            max_triangle_node = local_max_triangle_node;
            max_triangle_number = local_max_triangle_number;
          }
        }
      }
    }

    remove_node_k(max_triangle_node);
    remove_node(max_triangle_node);
    ans = get_nodes_k();

    struct timeval end;
    gettimeofday(&end, NULL);
    //#ifdef DEBUG
    printf("removing node %u with tnum %lu, time: %lf\n", max_triangle_node,
           max_triangle_number, (1000000.0 * (end.tv_sec - start.tv_sec) +
                                 end.tv_usec - start.tv_usec) /
                                    1000000.0);
    //#endif
  }

  restore_adj_list();
  this->graph_k(1);
  return ans;
}
