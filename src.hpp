#pragma once
#include "simulator.hpp"

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  Matrix *K_cum = nullptr;
  Matrix *V_cum = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Prepare K and V
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);

    if (i == 0) {
      K_cum = keys[0];
      V_cum = values[0];
    } else {
      Matrix *new_K = matrix_memory_allocator.Allocate("K_" + std::to_string(i));
      Matrix *new_V = matrix_memory_allocator.Allocate("V_" + std::to_string(i));
      gpu_sim.Concat(K_cum, keys[i], new_K, 0, kInSharedMemory);
      gpu_sim.Concat(V_cum, values[i], new_V, 0, kInSharedMemory);
      // Release old cum matrices if they were allocated dynamically
      if (i > 1) {
        gpu_sim.ReleaseMatrix(K_cum);
        gpu_sim.ReleaseMatrix(V_cum);
      }
      K_cum = new_K;
      V_cum = new_V;
    }

    // Transpose K_cum in-place
    gpu_sim.Transpose(K_cum, kInSharedMemory);

    // Q @ K^T -> QKt of shape (i+1, i+1)
    Matrix *QKt = matrix_memory_allocator.Allocate("QKt_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K_cum, QKt);

    // Transpose K_cum back for future concat
    gpu_sim.Transpose(K_cum, kInSharedMemory);

    // Softmax row-wise
    std::vector<Matrix *> softmax_rows;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *row = matrix_memory_allocator.Allocate("row_" + std::to_string(r));
      gpu_sim.GetRow(QKt, r, row, kInSharedMemory);

      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(r));
      gpu_sim.MatExp(row, exp_row);
      gpu_sim.ReleaseMatrix(row);

      Matrix *sum_val = matrix_memory_allocator.Allocate("sum_" + std::to_string(r));
      gpu_sim.Sum(exp_row, sum_val);

      Matrix *soft_row = matrix_memory_allocator.Allocate("soft_row_" + std::to_string(r));
      gpu_sim.MatDiv(exp_row, sum_val, soft_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_val);

      softmax_rows.push_back(soft_row);
    }

    gpu_sim.ReleaseMatrix(QKt);

    // Concat softmax rows
    Matrix *softmax_mat = softmax_rows[0];
    for (size_t r = 1; r <= i; ++r) {
      Matrix *next_concat = matrix_memory_allocator.Allocate("soft_concat_" + std::to_string(r));
      gpu_sim.Concat(softmax_mat, softmax_rows[r], next_concat, 0, kInSharedMemory);
      if (r > 1) {
        gpu_sim.ReleaseMatrix(softmax_mat);
      }
      gpu_sim.ReleaseMatrix(softmax_rows[r]);
      softmax_mat = next_concat;
    }

    // Softmax @ V -> ans
    Matrix *ans = matrix_memory_allocator.Allocate("ans_" + std::to_string(i));
    gpu_sim.MatMul(softmax_mat, V_cum, ans);

    if (i > 0) {
      gpu_sim.ReleaseMatrix(softmax_mat);
    }
    gpu_sim.ReleaseMatrix(current_query);

    // Move ans to HBM
    gpu_sim.MoveMatrixToGpuHbm(ans);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*ans);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
