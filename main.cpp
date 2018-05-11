#include <cstdio>
#include "graph.h"
int main(int argc, char** argv) {
  char* graph_filename;
  size_t k, n;
  if (argc != 4) {
    printf("usage:\n");
    printf("      %s [graph file name] [k] [n]\n", argv[0]);
    return -1;
  } else {
    graph_filename = argv[1];
    sscanf(argv[2], "%lu", &k);
    sscanf(argv[3], "%lu", &n);
  }

  graph g(graph_filename);
  auto ans = g.tera(k, n);

  return 0;
}
