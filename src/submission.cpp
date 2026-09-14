#include "submission.hpp"

Grid::Grid(std::size_t rows, std::size_t cols)
    : nodes(rows * cols), rows_(rows), cols_(cols) {}

std::tuple<std::size_t, std::size_t> Grid::get_dimensions() const {
  return std::tuple<std::size_t, std::size_t>(rows_, cols_);
}

double &Grid::operator()(std::size_t i, std::size_t j) {
  return nodes[i * cols_ + j];
}

double Grid::operator()(std::size_t i, std::size_t j) const {
  return nodes[i * cols_ + j];
}

void Grid::copy_boundary(const Grid &ref) {
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

void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  auto [rows, cols] = old_grid.get_dimensions();
  new_grid.copy_boundary(old_grid);

  if (rows < 3 || cols < 3)
    return;

  // __restrict promises src/dst don't alias, letting the compiler
  // vectorize the inner loop.
  const double *__restrict src = old_grid.data();
  double *__restrict dst = new_grid.data();

  // Interior rows are independent, so parallelize across them.
  #pragma omp parallel for
  for (std::size_t i = 1; i < rows - 1; ++i) {
    const double *__restrict srow = src + i * cols;
    const double *__restrict srowU = src + (i - 1) * cols;
    const double *__restrict srowD = src + (i + 1) * cols;
    double *__restrict drow = dst + i * cols;

    // 5-point stencil: center weighted 0.5, each of the four
    // neighbors (up/down/left/right) weighted 0.125.
    #pragma omp simd
    for (std::size_t j = 1; j < cols - 1; ++j) {
      drow[j] = 0.5 * srow[j] +
                0.125 * (srowU[j] + srowD[j] + srow[j - 1] + srow[j + 1]);
    }
  }
}
