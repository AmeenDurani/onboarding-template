#pragma once

#include <cstddef>
#include <tuple>
#include <vector>
#include <cstring>

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
};

inline Grid::Grid(std::size_t rows, std::size_t cols)
    : nodes(rows * cols), rows_(rows), cols_(cols) {}

inline std::tuple<std::size_t, std::size_t> Grid::get_dimensions() const {
  return std::tuple<std::size_t, std::size_t>(rows_, cols_);
}

inline double &Grid::operator()(std::size_t i, std::size_t j) {
  return nodes[i * cols_ + j];
}

inline double Grid::operator()(std::size_t i, std::size_t j) const {
  return nodes[i * cols_ + j];
}

// Copies the outer border (row 0, row rows_-1, col 0, col cols_-1) from
// ref into this grid, leaving the interior untouched.
inline void copy_boundary(const double *src, double *dst, std::size_t rows_,
                   std::size_t cols_) {

  // Use memcpy since memory is contiguous.
  memcpy(dst, src, cols_ * sizeof(double));
  memcpy(dst + (rows_ - 1) * cols_, src + (rows_ - 1) * cols_,
         cols_ * sizeof(double));

  // Iterate over rows for copying verticle boundaries.
  #pragma omp parallel for
  for (std::size_t i = 1; i < rows_ - 1; ++i) {
    const double *__restrict srow = src + i * cols_;
    double *__restrict drow = dst + i * cols_;

    drow[0] = srow[0];
    drow[cols_ - 1] = srow[cols_ - 1];
  }
}

// Computes grid nodes within the defined boundaries using the 2D heat diffusion
// equation.
inline void inner_compute(const double *src, double *dst, std::size_t rows_,
                          std::size_t cols_) {
  // Check if there are any interior nodes to perform operations on.
  if (rows_ < 3 || cols_ < 3)
    return;

  // Interior rows are independent, so parallelize across them.
  #pragma omp parallel for
  for (std::size_t i = 1; i < rows_ - 1; ++i) {
    const double *__restrict srow = src + i * cols_;
    const double *__restrict srowU = src + (i - 1) * cols_;
    const double *__restrict srowD = src + (i + 1) * cols_;
    double *__restrict drow = dst + i * cols_;

    // 5-point stencil: center weighted 0.5, each of the four
    // neighbors (up/down/left/right) weighted 0.125.
    #pragma omp simd
    for (std::size_t j = 1; j < cols_ - 1; ++j) {
      drow[j] = 0.5 * srow[j] +
                0.125 * (srowU[j] + srowD[j] + srow[j - 1] + srow[j + 1]);
    }
  }
}

// Computes new_grid from old_grid using a 5-point stencil over the
// interior, and copies old_grid's boundary into new_grid unchanged.
inline void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  auto [r, c] = old_grid.get_dimensions();

  // __restrict promises src/dst don't alias, letting the compiler
  // vectorize the inner loop.
  const double *__restrict src = old_grid.data();
  double *__restrict dst = new_grid.data();

  copy_boundary(src, dst, r, c);
  inner_compute(src, dst, r, c);
}
