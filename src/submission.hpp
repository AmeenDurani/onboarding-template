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

  std::vector<std::vector<double>> grid_;

public:
  Grid(std::size_t rows, std::size_t cols);

  std::tuple<std::size_t, std::size_t> get_grid_dims() const;

  double &operator()(std::size_t i, std::size_t j);
  double operator()(std::size_t i, std::size_t j) const;

  const std::vector<double> &operator()(std::size_t i) const;

  void copy_boundary(const Grid &ref);
};

Grid::Grid(std::size_t rows, std::size_t cols) {
  grid_ = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
  rows_ = rows;
  cols_ = cols;
}

std::tuple<std::size_t, std::size_t> Grid::get_grid_dims() const {
  return std::tuple<std::size_t, std::size_t>(rows_, cols_);
}

double &Grid::operator()(std::size_t i, std::size_t j) { return grid_[i][j]; }

double Grid::operator()(std::size_t i, std::size_t j) const {
  return grid_[i][j];
}

const std::vector<double> &Grid::operator()(std::size_t i) const {
  return grid_[i];
}

void Grid::copy_boundary(const Grid& ref) {
  // Top and Bottom
  grid_[0] = ref(0);
  grid_[rows_ - 1] = ref(rows_ - 1);

  // Row by Row
  for (std::size_t i = 1; i < rows_ - 1; ++i) {
    grid_[i][0] = ref(i, 0);
    grid_[i][cols_ - 1] = ref(i, cols_ - 1);
  }
}

void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  auto [rows, cols] = old_grid.get_grid_dims();
  if (rows < 3 || cols < 3)
    return;
  
  new_grid.copy_boundary(old_grid);

  for (std::size_t i = 1; i < rows - 1; ++i) {
    const auto &top = old_grid(i + 1);
    const auto &middle = old_grid(i);
    const auto &bottom = old_grid(i - 1);

    for (std::size_t j = 1; j < cols - 1; ++j) {
      new_grid(i, j) =
          0.5 * middle[j] +
          0.125 * (middle[j - 1] + middle[j + 1] + top[j] + bottom[j]);
    }
  }
}
