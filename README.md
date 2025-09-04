# Modular Decomposition in C (O(n+m))

Linear-time modular decomposition using LBFS + incremental assembly.

- **Language:** C11
- **Graph storage:** CSR (undirected; each edge stored twice)
- **Complexity:** Time `O(n+m)`, Space `O(n+m)`

## Build & Run

### Windows (MSYS2 MinGW-w64)
```bash
gcc modular_decomposition.c -std=c11 -O2 -Wall -Wextra -o md.exe
./md.exe
```

### Linux / macOS
```bash
gcc modular_decomposition.c -std=c11 -O2 -Wall -Wextra -o md
./md
```

### What it prints

main() constructs six small graphs (K4, K3,2, a tree, a nested example, C5, and K3+K2), prints the modular decomposition tree, then verify: True if each internal node is a module.

### Complexity

LBFS: each vertex/edge handled O(1) times with constant-time block ops -> O(n+m).

Assembly: each vertex scans its neighbors once, using epoch bits for cross/consistency checks -> O(m).

MD tree: ≤ 2n−1 nodes; dynamic arrays grow geometrically -> O(n).

### Example Output
```
== Clique4 ==
(0, 1, 2, 3) SERIES
  3
  2
  1
  0
verify: True

== K32 ==
(0, 1, 2, 3, 4) PRIME
  (1, 2) PARALLEL
    2
    1
  (3, 4) PARALLEL
    4
    3
  0
verify: True
```

### Notes

Tested with GCC/Clang. MSVC is not targeted.

Single-file implementation, no variable-length arrays in the final code.
