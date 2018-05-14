#include <algorithm>
#include <cstdio>
#include <iostream>
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

  std::sort(ans.begin(), ans.end());

  printf("candidates:\n");
  for (auto member : ans) {
    std::cout << member << " ";
  }
  std::cout << std::endl;

  return 0;
}
