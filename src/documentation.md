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

### Observation

In the first iteration, our operator function was as follows:

```cpp
double& Grid::operator()(std::size_t i, std::size_t j) {
  if (i >= rows_ || j >= cols_) throw std::out_of_range("Grid index out of bounds");
  return grid_[i][j];
}
```

For every element, recall that we'd be doing this check 5 times per interior node. This easily blows up when faced with increased node count. To optimize, ensure proper handling within the apply_stencil() function.

### Performance

| Run | Runtime (ms) | Score |
|---:|-------------:|------:|
| 1   | 515.413      | 0.611 |
| 2   | 399.500      | 0.564 |
| 3   | 396.338      | 0.562 |
| 4   | 527.738      | 0.441 |

## Optimization 3
### Observation

We don't actually need to copy the whole grid to satisfy the halo conditions (constant BCs). A whole bunch of computation is wasted for copying the old grid to the new one, when instead we can create another wrapper function to satisfy the conditions by iterating over the grid manually.

### Performance

| Run | Runtime (ms) | Score |
|---:|-------------:|------:|
| 1   | 177.214      | 0.935 |
| 2   | 176.190     | 1.010 |
| 3   | 176.958      | 1.021 |
| 4   | 188.581     | 0.966 |