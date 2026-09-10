#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <tuple>
#include <thread>

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

  double& operator()(std::size_t i, std::size_t j);
  double  operator()(std::size_t i, std::size_t j) const;

  const std::vector<double>& operator()(std::size_t i) const;
}; 

Grid::Grid(std::size_t rows, std::size_t cols) {
    grid_ = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
    rows_ = rows;
    cols_ = cols;
}

std::tuple<std::size_t, std::size_t> Grid::get_grid_dims() const {
    return std::tuple<std::size_t, std::size_t>(rows_, cols_);
}

double& Grid::operator()(std::size_t i, std::size_t j) {
  // if (i >= rows_ || j >= cols_) throw std::out_of_range("Grid index out of bounds");
  return grid_[i][j];
}

double Grid::operator()(std::size_t i, std::size_t j) const {
  // if (i >= rows_ || j >= cols_) throw std::out_of_range("Grid index out of bounds");
  return grid_[i][j];
}

const std::vector<double>& Grid::operator()(std::size_t i) const {
  return grid_[i];
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  auto[rows, cols] = old_grid.get_grid_dims();
  new_grid = old_grid;
  if (rows < 3 || cols < 3) return;

  

  for (std::size_t i = 1; i < rows - 1; ++i) {
    const auto& top = old_grid(i + 1);
    const auto& middle = old_grid(i);
    const auto& bottom = old_grid(i - 1);

    for (std::size_t j = 1; j < cols - 1; ++j) {
      new_grid(i, j) =  0.5   * middle[j] + 
                          0.125 * ( middle[j - 1] + 
                                    middle[j + 1] + 
                                    top[j] + 
                                    bottom[j]
                                  );
    }
  }
}
