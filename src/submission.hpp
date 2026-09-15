#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

constexpr std::size_t TILE_SIZE = 16;

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

inline void Grid::copy_boundary(const Grid &ref) {
  // Top and Bottom rows, full width.
  for (std::size_t i = 0; i < cols_; ++i) {
    nodes[i] = ref(0, i);
    nodes[(rows_ - 1) * cols_ + i] = ref(rows_ - 1, i);
  }

  // Left and right columns of the remaining rows; corners are already
  // covered by the top/bottom loop above, so skip row 0 and rows_ - 1.
  for (std::size_t i = 1; i < rows_ - 1; ++i) {
    nodes[i * cols_] = ref(i, 0);
    nodes[i * cols_ + cols_ - 1] = ref(i, cols_ - 1);
  }
}

// Computes new_grid from old_grid using a 5-point stencil over the
// interior, and copies old_grid's boundary into new_grid unchanged.
inline void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  auto [rows, cols] = old_grid.get_dimensions();
  new_grid.copy_boundary(old_grid);

  if (rows < 3 || cols < 3)
    return;

  // __restrict promises src/dst don't alias, letting the compiler
  // vectorize the inner loop.
  const double *__restrict src = old_grid.data();
  double *__restrict dst = new_grid.data();

  // Obtain max number of tile rows/cols.
  const std::size_t tile_rows = (rows - 2 + TILE_SIZE - 1) / TILE_SIZE;
  const std::size_t tile_cols = (cols - 2 + TILE_SIZE - 1) / TILE_SIZE;

  #pragma omp parallel for
  for (std::size_t tile = 0; tile < tile_rows * tile_cols; ++tile) {
    // Convert 1D tile index into a 2D tile coordinate.
    const std::size_t tile_i = tile / tile_cols;
    const std::size_t tile_j = tile % tile_cols;

    // Convert the tile coordinate into a grid coordinate.
    const std::size_t ii = 1 + tile_i * TILE_SIZE;
    const std::size_t jj = 1 + tile_j * TILE_SIZE;

    // Find the terminating row/col.
    const std::size_t i_end = std::min(ii + TILE_SIZE, rows - 1);
    const std::size_t j_end = std::min(jj + TILE_SIZE, cols - 1);

    for (std::size_t i = ii; i < i_end; ++i) {
      const double *__restrict srow = src + i * cols;
      const double *__restrict srowU = src + (i - 1) * cols;
      const double *__restrict srowD = src + (i + 1) * cols;
      double *__restrict drow = dst + i * cols;

      // 5-point stencil: center weighted 0.5, each of the four
      // neighbors (up/down/left/right) weighted 0.125.
      #pragma omp simd
      for (std::size_t j = jj; j < j_end; ++j) {
        drow[j] = 0.5 * srow[j] +
                  0.125 * (srowU[j] + srowD[j] + srow[j - 1] + srow[j + 1]);
      }
    }
  }
}
