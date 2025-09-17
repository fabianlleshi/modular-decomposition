# Modular Decomposition in C (O(n+m))

This repository provides a reference implementation of modular decomposition in linear time.

- Linear-time modular decomposition using LBFS + incremental assembly.
- **Language:** C11
- **Graph storage:** CSR (undirected; each edge stored twice)
- **Complexity:** Time `O(n+m)`, Space `O(n+m)`

## Installation

Clone the repository:
```bash
git clone https://github.com/fabianlleshi/modular-decomposition.git
cd modular-decomposition
```

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

main() constructs six small graphs (K4, K3,2, a tree, a nested example, C5, and K3+K2), prints the modular decomposition tree, then outputs `verify: True` if each internal node is a module.

## Complexity

LBFS: each vertex/edge handled O(1) times with constant-time block ops -> O(n+m).

Assembly: each vertex scans its neighbors once, using epoch bits for cross/consistency checks -> O(m).

MD tree: ≤ 2n−1 nodes; dynamic arrays grow geometrically -> O(n).

## Example Output
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

## Notes
- Tested with GCC/Clang.

- MSVC is not targeted.

- Single-file implementation, no variable-length arrays in the final code.

## License
This project is licensed under the [MIT License](LICENSE).


## Citing

If you find this implementation useful in academic work, please cite:

**APA**  
Lleshi, F. (2025). *Modular Decomposition in C (O(n+m))* [Computer software]. GitHub. [https://github.com/fabianlleshi/modular-decomposition](https://github.com/fabianlleshi/modular-decomposition)

**BibTeX**
```bibtex
@software{lleshi_modular_decomposition_2025,
  author  = {Fabian Lleshi},
  title   = {Modular Decomposition in C (O(n+m))},
  year    = {2025},
  url     = {https://github.com/fabianlleshi/modular-decomposition},
  note    = {Linear-time modular decomposition using LBFS + incremental assembly}
}
```
