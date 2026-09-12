#pragma once

#include <cstddef>
#include <functional>
#include <thread>
#include <tuple>
#include <vector>

static constexpr std::size_t groups = 10;

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
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
  std::vector<double> &operator()(std::size_t i);

  void copy_boundary(const Grid& ref);
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

std::vector<double> &Grid::operator()(std::size_t i) { return grid_[i]; }

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

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  auto [rows, cols] = old_grid.get_grid_dims();
  
  new_grid.copy_boundary(old_grid);
  
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
