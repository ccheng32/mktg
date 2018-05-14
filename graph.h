#include <unordered_map>
#include <unordered_set>
#include <vector>
typedef uint32_t node_t;

class graph {
 public:
  graph(char* filename);
  size_t num_nodes() const;
  std::vector<node_t> get_nodes();
  bool has_edge(node_t a, node_t b) const;
  std::vector<node_t> tera(size_t k, size_t n);
  size_t get_degree(node_t node) const;

 private:
  std::unordered_map<node_t, std::unordered_set<node_t>> adj_list;
  std::unordered_map<node_t, std::unordered_set<node_t>> adj_list_copy;
  void make_adj_list_copy();
  void restore_adj_list();
  void add_undirected_edge(node_t a, node_t b);
  void graph_k(size_t newk);
  std::vector<node_t> get_nodes_k();
  size_t triangle_number(node_t node);
  bool has_edge_k(node_t a, node_t b) const;
  void remove_node_k(node_t node);
  void remove_node(node_t node);

  size_t k;
  std::unordered_map<node_t, std::unordered_set<node_t>> adj_list_k;
};
