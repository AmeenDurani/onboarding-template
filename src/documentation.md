# Design Choices, Evaluation, Iterations

## Naive Attempt

### Design

This attempt focuses on making a basic solution without any optimizations applied.

**Task 1:** 2D Array, makes accessing specific grid elements relatively simple (i.e. grid[j][i]). Will need to handle BCs carefully.

**Task 2:** Copy the old grid to the new, then only modify the interior elements.

### Performance

The initial benchmark produced the following results:

| Run | Runtime (ms) | Score |
|---:|-------------:|------:|
| 1   | 880.793      | 0.295 |
| 2   | 758.849      | 0.457 |
| 3   | 887.474      | 0.271 |

The baseline is significantly slower than the provided reference implementation, with all three runs producing a score below 1. This establishes that the initial implementation requires performance improvements.

The relatively large variation between runs also suggests that individual benchmark runs should not be treated as exact measurements; subsequent optimizations will be evaluated using repeated runs and compared against this baseline.

## Optimization 1

### Motivation

In the first iteration, our operator function was as follows:

```cpp
double& Grid::operator()(std::size_t i, std::size_t j) {
  if (i >= rows_ || j >= cols_) throw std::out_of_range("Grid index out of bounds");
  return grid_[i][j];
}
```

For every element, recall that we'd be doing this check 5 times per interior node. This easily blows up when faced with increased node count. To optimize, ensure proper handling within the apply_stencil() function.

### Results

| Run | Runtime (ms) | Score |
|---:|-------------:|------:|
| 1   | 515.413      | 0.611 |
| 2   | 399.500      | 0.564 |
| 3   | 396.338      | 0.562 |
| 4   | 527.738      | 0.441 |

## Optimization 2

### Motivation

Computation is pretty much the same across every node, this optimization will investigate using parallel programming to compute new rows in parallel.

**Added functions**:

```cpp
void calculate_row(const std::vector<double> &top,
                   const std::vector<double> &middle,
                   const std::vector<double> &bottom,
                   std::vector<double> &output) {
  std::size_t cols = top.size();
  for (std::size_t j = 1; j < cols - 1; ++j) {
    output[j] = 0.5 * middle[j] +
                0.125 * (middle[j - 1] + middle[j + 1] + top[j] + bottom[j]);
  }
}

void calculate_row_group(const Grid &input_grid, Grid &output_grid,
                         std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; ++i) {
    calculate_row(input_grid(i + 1), input_grid(i), input_grid(i - 1),
                  output_grid(i));
  }
}
```

**Modified Stencil Code**:
```cpp
void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  auto [rows, cols] = old_grid.get_grid_dims();
  new_grid = old_grid;
  if (rows < 3 || cols < 3)
    return;

  std::vector<std::thread> threads;
  std::size_t interior_rows = rows - 2;
  std::size_t base = interior_rows / groups;
  std::size_t remainder = interior_rows % groups;

  std::size_t begin = 1;

  for (std::size_t g = 0; g < groups; ++g) {
    std::size_t chunk_size = base + (g < remainder ? 1 : 0);
    std::size_t end = begin + chunk_size;

    threads.emplace_back(calculate_row_group, std::cref(old_grid),
                         std::ref(new_grid), begin, end);

    begin = end;
  }

  for (auto &thread : threads) {
    thread.join();
  }
}
```
### Results

| Thread Count | Average Runtime (ms) | Average Score |
|---:|---:|---:|
| 1  | 387.377 | 0.486 |
| 10 | 334.851 | 0.569 |
| 16 | 332.414 | 0.581 |
