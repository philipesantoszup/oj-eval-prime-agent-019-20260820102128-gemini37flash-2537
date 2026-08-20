#pragma once
#include "simulator.hpp"

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  Matrix *K_T = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    size_t n = i + 1;
    auto current_query = rater.GetNextQuery(); // in HBM

    // Transpose keys[i] in HBM: shape becomes (512, 1)
    gpu_sim.Transpose(keys[i], kInGpuHbm);

    if (i == 0) {
      K_T = keys[0];
    } else {
      Matrix *new_KT = matrix_memory_allocator.Allocate();
      gpu_sim.Concat(K_T, keys[i], new_KT, 1, kInGpuHbm);
      if (i > 1) {
        gpu_sim.ReleaseMatrix(K_T);
      }
      K_T = new_KT;
    }

    // Outer product for Q @ K_T over 512 channels
    Matrix *QKt_acc = nullptr;
    for (size_t c = 0; c < 512; ++c) {
      Matrix *q_col = matrix_memory_allocator.Allocate();
      gpu_sim.GetColumn(current_query, c, q_col, kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(q_col);

      Matrix *kt_row = matrix_memory_allocator.Allocate();
      gpu_sim.GetRow(K_T, c, kt_row, kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(kt_row);

      Matrix *prod = matrix_memory_allocator.Allocate();
      gpu_sim.MatMul(q_col, kt_row, prod);
      gpu_sim.ReleaseMatrix(q_col);
      gpu_sim.ReleaseMatrix(kt_row);

      if (c == 0) {
        QKt_acc = prod;
      } else {
        Matrix *new_acc = matrix_memory_allocator.Allocate();
        gpu_sim.MatAdd(QKt_acc, prod, new_acc);
        gpu_sim.ReleaseMatrix(QKt_acc);
        gpu_sim.ReleaseMatrix(prod);
        QKt_acc = new_acc;
      }
    }

    gpu_sim.ReleaseMatrix(current_query);

    // Softmax row-wise in Shared Memory
    std::vector<Matrix *> softmax_rows;
    for (size_t r = 0; r < n; ++r) {
      Matrix *row = matrix_memory_allocator.Allocate();
      gpu_sim.GetRow(QKt_acc, r, row, kInSharedMemory);

      Matrix *exp_row = matrix_memory_allocator.Allocate();
      gpu_sim.MatExp(row, exp_row);
      gpu_sim.ReleaseMatrix(row);

      Matrix *sum_val = matrix_memory_allocator.Allocate();
      gpu_sim.Sum(exp_row, sum_val);

      Matrix *soft_row = matrix_memory_allocator.Allocate();
      gpu_sim.MatDiv(exp_row, sum_val, soft_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_val);

      softmax_rows.push_back(soft_row);
    }
    gpu_sim.ReleaseMatrix(QKt_acc);

    // Compute Ans row by row:
    // For row r: softmax_rows[r] is shape (1, n).
    // ans_row_r = softmax_rows[r] @ V
    // = sum_{k=0}^{n-1} softmax_rows[r][k] * values[k]
    std::vector<Matrix *> ans_rows;
    for (size_t r = 0; r < n; ++r) {
      Matrix *ans_row_acc = nullptr;
      for (size_t k = 0; k < n; ++k) {
        Matrix *s_elem = matrix_memory_allocator.Allocate();
        gpu_sim.GetColumn(softmax_rows[r], k, s_elem, kInSharedMemory); // (1, 1)

        gpu_sim.MoveMatrixToSharedMem(values[k]); // (1, 512)

        Matrix *prod = matrix_memory_allocator.Allocate();
        gpu_sim.MatMul(s_elem, values[k], prod); // (1, 512)
        gpu_sim.ReleaseMatrix(s_elem);
        gpu_sim.MoveMatrixToGpuHbm(values[k]);

        if (k == 0) {
          ans_row_acc = prod;
        } else {
          Matrix *new_acc = matrix_memory_allocator.Allocate();
          gpu_sim.MatAdd(ans_row_acc, prod, new_acc);
          gpu_sim.ReleaseMatrix(ans_row_acc);
          gpu_sim.ReleaseMatrix(prod);
          ans_row_acc = new_acc;
        }
      }
      gpu_sim.ReleaseMatrix(softmax_rows[r]);

      // Move ans_row_acc to HBM
      gpu_sim.MoveMatrixToGpuHbm(ans_row_acc);
      ans_rows.push_back(ans_row_acc);
    }

    // Assemble Ans in HBM by concatenating ans_rows vertically
    Matrix *ans = ans_rows[0];
    for (size_t r = 1; r < n; ++r) {
      Matrix *next_ans = matrix_memory_allocator.Allocate();
      gpu_sim.Concat(ans, ans_rows[r], next_ans, 0, kInGpuHbm);
      if (r > 1) {
        gpu_sim.ReleaseMatrix(ans);
      }
      gpu_sim.ReleaseMatrix(ans_rows[r]);
      ans = next_ans;
    }

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
