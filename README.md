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

Complexity (sketch)

LBFS: each vertex/edge handled O(1) times with constant-time block ops → O(n+m).

Assembly: each vertex scans its neighbors once, using epoch bits for cross/consistency checks → O(m).

MD tree: ≤ 2n−1 nodes; dynamic arrays grow geometrically → O(n).

### Notes

Tested with GCC/Clang. MSVC is not targeted.

Single-file implementation; no variable-length arrays in the final code.