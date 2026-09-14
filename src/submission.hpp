#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

// 2D grid of doubles stored as a single row-major buffer.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;

  std::vector<double> nodes;

public:
  Grid(std::size_t rows, std::size_t cols);

  // Returns (rows, cols).
  std::tuple<std::size_t, std::size_t> get_dimensions() const;

  // Row-major element access: nodes[i * cols_ + j].
  double &operator()(std::size_t i, std::size_t j);
  double operator()(std::size_t i, std::size_t j) const;

  // Raw access to the underlying buffer, e.g. for pointer-based/SIMD code.
  double *data() { return nodes.data(); }
  const double *data() const { return nodes.data(); }

  // Copies the outer border (row 0, row rows_-1, col 0, col cols_-1) from
  // ref into this grid, leaving the interior untouched.
  void copy_boundary(const Grid &ref);
};

// Computes new_grid from old_grid using a 5-point stencil over the
// interior, and copies old_grid's boundary into new_grid unchanged.
void apply_stencil(const Grid &old_grid, Grid &new_grid);

