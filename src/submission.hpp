#pragma once

#include <cstddef>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;

  std::vector<double> nodes;

public:
  Grid(std::size_t rows, std::size_t cols);
  std::tuple<std::size_t, std::size_t> get_dimensions() const;

  double &operator()(std::size_t i, std::size_t j);
  double operator()(std::size_t i, std::size_t j) const;

  double *data() { return nodes.data(); }
  const double *data() const { return nodes.data(); }

  void copy_boundary(const Grid &ref);
};

Grid::Grid(std::size_t rows, std::size_t cols) {
  nodes = std::vector<double>(rows * cols);
  rows_ = rows;
  cols_ = cols;
}

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
  // Top and Bottom
  for (std::size_t i = 0; i < cols_; ++i) {
    nodes[i] = ref(0, i);
    nodes[(rows_ - 1) * cols_ + i] = ref(rows_ - 1, i);
  }

  // Row by Row
  for (std::size_t i = 1; i < rows_ - 1; ++i) {
    nodes[i * cols_] = ref(i, 0);
    nodes[i * cols_ + cols_ - 1] = ref(i, cols_ - 1);
  }
}

void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  auto [rows, cols] = old_grid.get_dimensions();
  if (rows < 3 || cols < 3)
    return;

  new_grid.copy_boundary(old_grid);

  const double *__restrict src = old_grid.data();
  double *__restrict dst = new_grid.data();

  for (std::size_t i = 1; i < rows - 1; ++i) {
    const double *__restrict srow = src + i * cols;
    const double *__restrict srowU = src + (i - 1) * cols;
    const double *__restrict srowD = src + (i + 1) * cols;
    double *__restrict drow = dst + i * cols;

    for (std::size_t j = 1; j < cols - 1; ++j) {
      drow[j] = 0.5 * srow[j] +
                0.125 * (srowU[j] + srowD[j] + srow[j - 1] + srow[j + 1]);
    }
  }
}
