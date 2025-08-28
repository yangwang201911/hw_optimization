// DPP OpenCL Host Code
// Test implementation for the DPP kernel functions

#include <iostream>
#include <vector>
#include <CL/opencl.hpp>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

class DPPOpenCL {
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Device device;
    
    // Kernels
    cl::Kernel kernel_init_diagonal;
    cl::Kernel kernel_batch_process;
    cl::Kernel kernel_test;

public:
    DPPOpenCL() {
        // Initialize OpenCL
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found");
        }
        
        // Get GPU device
        std::vector<cl::Device> devices;
        for (auto& platform : platforms) {
            std::vector<cl::Device> platform_devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &platform_devices);
            devices.insert(devices.end(), platform_devices.begin(), platform_devices.end());
        }
        
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices found");
        }
        
        device = devices[0];
        context = cl::Context({device});
        queue = cl::CommandQueue(context, device);
        
        std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }
    
    void loadKernel(const std::string& filename) {
        // Read kernel source
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open kernel file: " + filename);
        }
        
        std::stringstream ss;
        ss << file.rdbuf();
        std::string source = ss.str();
        
        // Compile program
        cl::Program::Sources sources;
        sources.push_back({source.c_str(), source.length()});
        
        program = cl::Program(context, sources);
        if (program.build({device}) != CL_SUCCESS) {
            std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cerr << "Build failed: " << build_log << std::endl;
            throw std::runtime_error("Kernel compilation failed");
        }
        
        // Create kernels
        kernel_init_diagonal = cl::Kernel(program, "dpp_init_diagonal");
        kernel_batch_process = cl::Kernel(program, "dpp_batch_process");
        kernel_test = cl::Kernel(program, "dpp_test_kernel");
        
        std::cout << "Kernels loaded successfully!" << std::endl;
    }
    
    std::vector<std::vector<int>> selectTokens(const std::vector<float>& kernel_data,
                                              int batch_size, int N, int T) {
        // Create OpenCL buffers
        cl::Buffer buffer_kernel_matrix(context, CL_MEM_READ_ONLY, sizeof(float) * kernel_data.size());
        cl::Buffer buffer_di2s(context, CL_MEM_READ_WRITE, sizeof(float) * N);
        cl::Buffer buffer_cis(context, CL_MEM_READ_WRITE, sizeof(float) * N * T);
        cl::Buffer buffer_selected(context, CL_MEM_WRITE_ONLY, sizeof(int) * T);
        cl::Buffer buffer_mask(context, CL_MEM_READ_WRITE, sizeof(int) * N);

        // Copy input data to device
        queue.enqueueWriteBuffer(buffer_kernel_matrix, CL_TRUE, 0, sizeof(float) * kernel_data.size(), kernel_data.data());        // Set kernel arguments
        kernel_batch_process.setArg(0, buffer_kernel_matrix);
        kernel_batch_process.setArg(1, buffer_di2s);
        kernel_batch_process.setArg(2, buffer_cis);
        kernel_batch_process.setArg(3, buffer_selected);
        kernel_batch_process.setArg(4, buffer_mask);
        kernel_batch_process.setArg(5, cl::Local(sizeof(float) * 256));  // local_values
        kernel_batch_process.setArg(6, cl::Local(sizeof(int) * 256));    // local_indices
        kernel_batch_process.setArg(7, batch_size);
        kernel_batch_process.setArg(8, N);
        kernel_batch_process.setArg(9, T);
        kernel_batch_process.setArg(10, 1e-8f);  // numerical_threshold
        
        // Execute kernel
        cl::NDRange global_size(batch_size * 256);  // 256 work items per batch
        cl::NDRange local_size(256);
        
        auto start = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(kernel_batch_process, cl::NullRange, global_size, local_size);
        queue.finish();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "OpenCL DPP execution time: " << duration.count() << " microseconds" << std::endl;
        
        // Read results
        std::vector<int> selected_flat(batch_size * T);
        queue.enqueueReadBuffer(buffer_selected, CL_TRUE, 0, sizeof(int) * batch_size * T, selected_flat.data());
        
        // Convert to 2D result
        std::vector<std::vector<int>> result(batch_size);
        for (int b = 0; b < batch_size; b++) {
            result[b].resize(T);
            for (int t = 0; t < T; t++) {
                result[b][t] = selected_flat[b * T + t];
            }
        }
        
        return result;
    }
    
    void testSimpleKernel() {
        std::cout << "\n=== Testing simple kernel ===" << std::endl;
        
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        std::vector<float> output(input.size());
        
        cl::Buffer buffer_in(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());
        cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());
        
        queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(float) * input.size(), input.data());
        
        kernel_test.setArg(0, buffer_in);
        kernel_test.setArg(1, buffer_out);
        kernel_test.setArg(2, (int)input.size());
        
        queue.enqueueNDRangeKernel(kernel_test, cl::NullRange, cl::NDRange(input.size()), cl::NullRange);
        queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(float) * output.size(), output.data());
        
        std::cout << "Input:  ";
        for (float v : input) std::cout << v << " ";
        std::cout << "\nOutput: ";
        for (float v : output) std::cout << v << " ";
        std::cout << std::endl;
    }
};

// Reference CPU implementation for comparison
std::vector<std::vector<int>> dppCPUReference(const std::vector<float>& kernel_data,
                                             int batch_size, int N, int T) {
    std::vector<std::vector<int>> results(batch_size);
    
    for (int b = 0; b < batch_size; b++) {
        std::vector<float> di2s(N);
        std::vector<bool> selected(N, false);
        std::vector<int> selected_indices;
        
        // Initialize diagonal elements
        for (int i = 0; i < N; i++) {
            int idx = b * N * N + i * N + i;
            di2s[i] = kernel_data[idx];
        }
        
        // Greedy selection
        for (int t = 0; t < T; t++) {
            // Find best token
            int best_idx = -1;
            float best_value = -std::numeric_limits<float>::infinity();
            
            for (int i = 0; i < N; i++) {
                if (!selected[i] && di2s[i] > best_value) {
                    best_value = di2s[i];
                    best_idx = i;
                }
            }
            
            if (best_idx == -1) break;
            
            selected_indices.push_back(best_idx);
            selected[best_idx] = true;
            
            // Simplified update (this is a basic approximation)
            for (int j = 0; j < N; j++) {
                if (!selected[j]) {
                    di2s[j] *= 0.9f;  // Simple decay approximation
                }
            }
        }
        
        results[b] = selected_indices;
    }
    
    return results;
}

int main(int argc, char** argv) {
    try {
        std::cout << "=== DPP OpenCL Test ===" << std::endl;
        
        // Initialize OpenCL
        DPPOpenCL dpp_ocl;
        
        // Load kernel
        std::string kernel_file = "dpp_kernel.cl";
        if (argc > 1) {
            kernel_file = argv[1];
        }
        
        std::cout << "\n=== Loading kernel file ===" << std::endl;
        dpp_ocl.loadKernel(kernel_file);

        // Test simple kernel first
        std::cout << "\n=== Test Simple Kernel ===" << std::endl;
        dpp_ocl.testSimpleKernel();
        
        // Test DPP algorithm
        std::cout << "\n=== Testing DPP Algorithm ===" << std::endl;
        
        const int batch_size = 1;
        const int N = 4;  // Number of tokens
        const int T = 2;  // Tokens to select
        
        // Create test kernel matrix (4x4)
        std::vector<float> kernel_data = {
            // Batch 0
            0.8f, 0.3f, 0.1f, 0.2f,  // token 0 row
            0.3f, 0.9f, 0.4f, 0.1f,  // token 1 row
            0.1f, 0.4f, 0.7f, 0.5f,  // token 2 row
            0.2f, 0.1f, 0.5f, 0.6f   // token 3 row
        };
        
        std::cout << "Input kernel matrix [" << batch_size << "x" << N << "x" << N << "]:" << std::endl;
        for (int i = 0; i < N; i++) {
            std::cout << "  [";
            for (int j = 0; j < N; j++) {
                std::cout << std::fixed << std::setprecision(1) << kernel_data[i * N + j];
                if (j < N-1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Run OpenCL implementation
        auto ocl_result = dpp_ocl.selectTokens(kernel_data, batch_size, N, T);
        
        // Run CPU reference
        auto start = std::chrono::high_resolution_clock::now();
        auto cpu_result = dppCPUReference(kernel_data, batch_size, N, T);
        auto end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "CPU reference time: " << cpu_duration.count() << " microseconds" << std::endl;
        
        // Print results
        std::cout << "\nResults comparison:" << std::endl;
        for (int b = 0; b < batch_size; b++) {
            std::cout << "Batch " << b << ":" << std::endl;
            std::cout << "  OpenCL: [";
            for (size_t i = 0; i < ocl_result[b].size(); i++) {
                std::cout << ocl_result[b][i];
                if (i < ocl_result[b].size()-1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            std::cout << "  CPU:    [";
            for (size_t i = 0; i < cpu_result[b].size(); i++) {
                std::cout << cpu_result[b][i];
                if (i < cpu_result[b].size()-1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "\n=== Test completed ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
