// DPP (Determinantal Point Process) OpenCL Kernel Implementation
// Based on TraditionalFastGreedyDPP algorithm
// Author: AI Assistant
// Date: 2025-08-28

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

// ==================== Utility Functions ====================

/**
 * @brief Atomic float compare and exchange for finding maximum
 */
inline void atomic_max_float(__global float* target, float value) {
    union {
        unsigned int as_uint;
        float as_float;
    } old_val, new_val;
    
    do {
        old_val.as_float = *target;
        new_val.as_float = max(old_val.as_float, value);
    } while (atomic_cmpxchg((__global unsigned int*)target, old_val.as_uint, new_val.as_uint) != old_val.as_uint);
}

/**
 * @brief Find argmax across a work group using reduction
 */
void workgroup_argmax(__global const float* data, 
                     __local float* local_values,
                     __local int* local_indices,
                     int size, 
                     __local int* result_idx) {
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);
    const int global_id = get_global_id(0);
    
    // Initialize local memory
    if (global_id < size && data[global_id] != -INFINITY) {
        local_values[local_id] = data[global_id];
        local_indices[local_id] = global_id;
    } else {
        local_values[local_id] = -INFINITY;
        local_indices[local_id] = -1;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction to find maximum
    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            if (local_values[local_id + stride] > local_values[local_id]) {
                local_values[local_id] = local_values[local_id + stride];
                local_indices[local_id] = local_indices[local_id + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        *result_idx = local_indices[0];
    }
}

// ==================== DPP Core Kernels ====================

/**
 * @brief Initialize diagonal elements (marginal gains) from kernel matrix
 * @param kernel_matrix Input kernel matrix [batch_size, N, N]
 * @param di2s Output diagonal elements [batch_size, N]
 * @param batch_idx Batch index to process
 * @param N Number of tokens
 */
__kernel void dpp_init_diagonal(__global const float* kernel_matrix,
                               __global float* di2s,
                               const int batch_idx,
                               const int N) {
    const int token_id = get_global_id(0);
    
    if (token_id >= N) return;
    
    // Extract diagonal element: kernel_matrix[batch_idx, token_id, token_id]
    int kernel_idx = batch_idx * N * N + token_id * N + token_id;
    di2s[batch_idx * N + token_id] = kernel_matrix[kernel_idx];
}

/**
 * @brief Find the token with maximum marginal gain (argmax operation)
 * @param di2s Marginal gains [batch_size, N]
 * @param selected_mask Mask of already selected tokens [batch_size, N]
 * @param batch_idx Batch index
 * @param N Number of tokens
 * @param result_idx Output: index of maximum token
 */
__kernel void dpp_find_best_token(__global const float* di2s,
                                 __global const int* selected_mask,
                                 __local float* local_values,
                                 __local int* local_indices,
                                 const int batch_idx,
                                 const int N,
                                 __global int* result_idx) {
    const int token_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);
    
    // Load data for this work group
    float value = -INFINITY;
    int idx = -1;
    
    if (token_id < N) {
        int data_idx = batch_idx * N + token_id;
        // Only consider tokens that haven't been selected
        if (selected_mask[data_idx] == 0) {
            value = di2s[data_idx];
            idx = token_id;
        }
    }
    
    local_values[local_id] = value;
    local_indices[local_id] = idx;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction to find maximum
    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            if (local_values[local_id + stride] > local_values[local_id]) {
                local_values[local_id] = local_values[local_id + stride];
                local_indices[local_id] = local_indices[local_id + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result from first work item in group
    if (local_id == 0 && local_indices[0] >= 0) {
        atomic_max(result_idx, local_indices[0]);
    }
}

/**
 * @brief Update orthogonal vector using Gram-Schmidt orthogonalization
 * @param kernel_matrix Input kernel matrix [batch_size, N, N]
 * @param cis Orthogonalized vectors [T, N]
 * @param di2s Marginal gains [batch_size, N]
 * @param batch_idx Batch index
 * @param selected_idx Index of selected token
 * @param iteration Current iteration number
 * @param N Number of tokens
 * @param numerical_threshold Numerical stability threshold
 */
__kernel void dpp_update_orthogonal_vector(__global const float* kernel_matrix,
                                         __global float* cis,
                                         __global const float* di2s,
                                         const int batch_idx,
                                         const int selected_idx,
                                         const int iteration,
                                         const int N,
                                         const float numerical_threshold) {
    const int token_id = get_global_id(0);
    
    if (token_id >= N) return;
    
    // Get normalization factor: sqrt(di2s[selected_idx])
    int di2s_idx = batch_idx * N + selected_idx;
    float norm_factor = sqrt(di2s[di2s_idx] + numerical_threshold);
    float inv_norm = 1.0f / norm_factor;
    
    // Copy kernel row: kernel_matrix[batch_idx, selected_idx, token_id]
    int kernel_idx = batch_idx * N * N + selected_idx * N + token_id;
    float eis_value = kernel_matrix[kernel_idx];
    
    // Subtract projections from previous orthogonal vectors
    for (int prev_t = 0; prev_t < iteration; prev_t++) {
        int cis_prev_selected_idx = prev_t * N + selected_idx;
        int cis_prev_token_idx = prev_t * N + token_id;
        
        float cis_sel = cis[cis_prev_selected_idx];
        float cis_token = cis[cis_prev_token_idx];
        
        eis_value -= cis_sel * cis_token;
    }
    
    // Normalize and store in cis matrix
    int cis_out_idx = iteration * N + token_id;
    cis[cis_out_idx] = eis_value * inv_norm;
}

/**
 * @brief Update marginal gains after selecting a token
 * @param cis Orthogonalized vectors [T, N]
 * @param di2s Marginal gains [batch_size, N] (input/output)
 * @param batch_idx Batch index
 * @param iteration Current iteration number
 * @param N Number of tokens
 */
__kernel void dpp_update_marginal_gains(__global const float* cis,
                                      __global float* di2s,
                                      const int batch_idx,
                                      const int iteration,
                                      const int N) {
    const int token_id = get_global_id(0);
    
    if (token_id >= N) return;
    
    int di2s_idx = batch_idx * N + token_id;
    
    // Skip if token is already selected (marked as negative infinity)
    if (di2s[di2s_idx] == -INFINITY) return;
    
    // Get the orthogonal component for this token
    int cis_idx = iteration * N + token_id;
    float eis_j = cis[cis_idx];
    
    // Update marginal gain: di2s[token_id] -= eis_j^2
    di2s[di2s_idx] -= eis_j * eis_j;
}

/**
 * @brief Mark selected token to prevent re-selection
 * @param di2s Marginal gains [batch_size, N] (input/output)
 * @param selected_mask Selection mask [batch_size, N] (input/output)
 * @param batch_idx Batch index
 * @param selected_idx Index of selected token
 * @param N Number of tokens
 */
__kernel void dpp_mark_selected_token(__global float* di2s,
                                    __global int* selected_mask,
                                    const int batch_idx,
                                    const int selected_idx,
                                    const int N) {
    const int token_id = get_global_id(0);
    
    if (token_id != selected_idx || token_id >= N) return;
    
    int idx = batch_idx * N + token_id;
    di2s[idx] = -INFINITY;
    selected_mask[idx] = 1;
}

// ==================== Batch Processing Kernels ====================

/**
 * @brief Process multiple batches in parallel
 * @param kernel_matrix Input kernel matrix [batch_size, N, N]
 * @param di2s Marginal gains workspace [batch_size, N]
 * @param cis Orthogonalized vectors workspace [batch_size, T, N]
 * @param selected_indices Output selected indices [batch_size, T]
 * @param selected_mask Selection mask workspace [batch_size, N]
 * @param batch_size Number of batches
 * @param N Number of tokens per batch
 * @param T Number of tokens to select
 * @param numerical_threshold Numerical stability threshold
 */
__kernel void dpp_batch_process(__global const float* kernel_matrix,
                              __global float* di2s,
                              __global float* cis,
                              __global int* selected_indices,
                              __global int* selected_mask,
                              __local float* local_values,
                              __local int* local_indices,
                              const int batch_size,
                              const int N,
                              const int T,
                              const float numerical_threshold) {
    const int batch_idx = get_group_id(0);
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);
    
    if (batch_idx >= batch_size) return;
    
    // Initialize diagonal elements for this batch
    for (int token_id = local_id; token_id < N; token_id += local_size) {
        int kernel_idx = batch_idx * N * N + token_id * N + token_id;
        int di2s_idx = batch_idx * N + token_id;
        di2s[di2s_idx] = kernel_matrix[kernel_idx];
        selected_mask[di2s_idx] = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    // Main DPP selection loop
    for (int t = 0; t < T; t++) {
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
        // Find best token for this iteration
        int best_idx = -1;
        float best_value = -INFINITY;
        
        // Each work item checks some tokens
        for (int token_id = local_id; token_id < N; token_id += local_size) {
            int di2s_idx = batch_idx * N + token_id;
            if (selected_mask[di2s_idx] == 0) {
                // Accept any unselected token, prioritizing higher gains
                if (best_idx == -1 || di2s[di2s_idx] > best_value) {
                    best_value = di2s[di2s_idx];
                    best_idx = token_id;
                }
            }
        }
        
        // Store in local memory for reduction
        local_values[local_id] = best_value;
        local_indices[local_id] = best_idx;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Reduction to find global maximum for this batch
        for (int stride = local_size / 2; stride > 0; stride /= 2) {
            if (local_id < stride) {
                // Prefer valid tokens (idx != -1), then higher values
                if (local_indices[local_id + stride] != -1 && 
                    (local_indices[local_id] == -1 || local_values[local_id + stride] > local_values[local_id])) {
                    local_values[local_id] = local_values[local_id + stride];
                    local_indices[local_id] = local_indices[local_id + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Get the selected token index
        int selected_idx = local_indices[0];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Safety check: if no valid token found, break early
        if (selected_idx == -1) {
            if (local_id == 0) {
                printf("Warning: No valid token found at iteration %d\n", t);
            }
            break;
        }
        
        // Store selected index
        if (local_id == 0) {
            selected_indices[batch_idx * T + t] = selected_idx;
        }
        
        // Calculate norm_factor BEFORE marking token as selected
        float current_gain = local_values[0];
        // Ensure positive value for sqrt
        float norm_factor = sqrt(max(current_gain, numerical_threshold));
        float inv_norm = 1.0f / norm_factor;
        
        // Mark token as selected
        if (local_id == 0) {
            int di2s_idx = batch_idx * N + selected_idx;
            selected_mask[di2s_idx] = 1;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
        // Update orthogonal vector
        for (int token_id = local_id; token_id < N; token_id += local_size) {
            // Copy kernel row
            int kernel_idx = batch_idx * N * N + selected_idx * N + token_id;
            float eis_value = kernel_matrix[kernel_idx];
            
            // Subtract projections from previous orthogonal vectors
            for (int prev_t = 0; prev_t < t; prev_t++) {
                int cis_prev_selected_idx = batch_idx * T * N + prev_t * N + selected_idx;
                int cis_prev_token_idx = batch_idx * T * N + prev_t * N + token_id;
                
                float cis_sel = cis[cis_prev_selected_idx];
                float cis_token = cis[cis_prev_token_idx];
                
                eis_value -= cis_sel * cis_token;
            }
            
            // Normalize and store
            int cis_out_idx = batch_idx * T * N + t * N + token_id;
            cis[cis_out_idx] = eis_value * inv_norm;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
        // Update marginal gains
        for (int token_id = local_id; token_id < N; token_id += local_size) {
            int di2s_idx = batch_idx * N + token_id;
            
            // Skip if already selected
            if (selected_mask[di2s_idx] == 1) continue;
            
            // Get orthogonal component
            int cis_idx = batch_idx * T * N + t * N + token_id;
            float eis_j = cis[cis_idx];
            
            // Update marginal gain
            di2s[di2s_idx] -= eis_j * eis_j;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    
    // Output selected token IDs after DPP algorithm completion
    if (batch_idx == 0 && local_id == 0) {
        printf("DPP Selection Results for Batch %d: [", batch_idx);
        int display_count = min(T, 10);
        for (int t = 0; t < display_count; t++) {
            printf("%d", selected_indices[batch_idx * T + t]);
            if (t < display_count-1) printf(", ");
        }
        if (T > 10) {
            printf(", +%d more", T - 10);
        }
        printf("]\n");
    }
}

// ==================== Simple Test Kernel ====================

/**
 * @brief Simple test kernel for debugging
 */
__kernel void dpp_test_kernel(__global const float* input,
                            __global float* output,
                            const int size) {
    const int idx = get_global_id(0);
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}
