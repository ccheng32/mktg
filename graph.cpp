#include "graph.h"
#include <cstdio>
#include <queue>

static void add_edge_gen(
    node_t a, node_t b,
    std::unordered_map<node_t, std::unordered_set<node_t>>& adj_list) {
  auto got = adj_list.find(a);
  if (got == adj_list.end()) {
    adj_list.insert(std::make_pair(a, std::unordered_set<node_t>({b})));
  } else {
    got->second.insert(b);
  }
}

void graph::make_adj_list_copy() { adj_list_copy = adj_list; }

void graph::restore_adj_list() { adj_list = adj_list = adj_list_copy; }

void graph::add_undirected_edge(node_t a, node_t b) {
  add_edge_gen(a, b, this->adj_list);
  add_edge_gen(b, a, this->adj_list);
}

static void add_undirected_edge_k(
    node_t a, const std::vector<node_t>& b,
    std::unordered_map<node_t, std::unordered_set<node_t>>& adj_list_k) {
  auto center_node = adj_list_k.find(a);
  for (node_t next : b) {
    center_node->second.insert(next);
  }
}

size_t graph::get_degree(node_t node) const {
  auto iter = adj_list.find(node);
  return iter == adj_list.end() ? 0 : iter->second.size();
}

void remove_node_gen(
    node_t node,
    std::unordered_map<node_t, std::unordered_set<node_t>>& adj_list) {
  adj_list.erase(node);
  for (auto iter = adj_list.begin(); iter != adj_list.end(); iter++) {
    iter->second.erase(node);
  }
  return;
}

void graph::remove_node_k(node_t node) { remove_node_gen(node, adj_list_k); }

void graph::remove_node(node_t node) { remove_node_gen(node, adj_list); }

std::vector<node_t> get_nodes_gen(
    std::unordered_map<node_t, std::unordered_set<node_t>> adj_list) {
  std::vector<node_t> v_nodes(adj_list.size());
  size_t i = 0;
  for (auto iter = adj_list.cbegin(); iter != adj_list.cend(); iter++) {
    v_nodes[i] = iter->first;
    i++;
  }
  return v_nodes;
}

std::vector<node_t> graph::get_nodes() { return get_nodes_gen(adj_list); }

std::vector<node_t> graph::get_nodes_k() { return get_nodes_gen(adj_list_k); }

graph::graph(char* filename) {
  FILE* graph_file = fopen(filename, "r");

  node_t a, b;
  while (fscanf(graph_file, "%d %d", &a, &b) == 2) {
    // construct graph here
    add_undirected_edge(a, b);
  }

  // k-graph
  k = 1;
  adj_list_k = adj_list;

  fclose(graph_file);
}

static bool has_edge_gen(
    node_t a, node_t b,
    const std::unordered_map<node_t, std::unordered_set<node_t>>& adj_list) {
  auto got = adj_list.find(a);
  if (got == adj_list.end() || got->second.find(b) == got->second.end()) {
    return false;
  } else {
    return true;
  }
}

bool graph::has_edge(node_t a, node_t b) const {
  return has_edge_gen(a, b, adj_list);
}

bool graph::has_edge_k(node_t a, node_t b) const {
  return has_edge_gen(a, b, adj_list_k);
}

void graph::graph_k(size_t newk) {
  if (newk == this->k) {
    return;
  } else {
    adj_list_k = adj_list;
    this->k = newk;
  }

  std::vector<node_t> nodes = get_nodes();

#pragma omp parallel
  {
    std::unordered_map<node_t, node_t> node_dist;
    std::queue<node_t> bfs_q;
    std::unordered_map<node_t, std::vector<node_t>> edges;

#pragma omp for
    for (auto iter = nodes.cbegin(); iter < nodes.cend(); iter++) {
      node_t center_node = *iter;

      // find all nodes that are within k hops from center node
      bfs_q.push(center_node);
      node_dist.insert(std::make_pair(center_node, 0));
      edges.insert(std::make_pair(center_node, std::vector<node_t>()));
      while (!bfs_q.empty()) {
        node_t curr_node = bfs_q.front();
        bfs_q.pop();
        if (node_dist[curr_node] >= newk) {
          break;
        }

        // traverse to next neighbors
        for (node_t neigh : adj_list[curr_node]) {
          if (node_dist.find(neigh) == node_dist.end()) {
            node_dist.insert(std::make_pair(neigh, node_dist[curr_node] + 1));
            bfs_q.push(neigh);
            edges[center_node].push_back(neigh);
          }
        }
      }

      std::queue<node_t>().swap(bfs_q);
      node_dist.clear();

#ifdef DEBUG
//  printf("node %d expanded\n", center_node);
#endif
    }

#pragma omp critical
    {
#ifdef DEBUG
      printf("%lu edges to add\n", edges.size());
#endif
      for (auto edge = edges.cbegin(); edge != edges.cend(); edge++) {
        add_undirected_edge_k(edge->first, edge->second, adj_list_k);
      }
    }
  }
  return;
}

size_t graph::num_nodes() const { return adj_list.size(); }
