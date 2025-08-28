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
#include <random>
#include <getopt.h>

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
        std::cout << "OpenCL DPP execution time: " << duration.count() << " us" << std::endl;
        
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

// Function to generate a random positive definite kernel matrix
std::vector<float> generateKernelMatrix(int N, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.1f, 1.0f);
    
    std::vector<float> kernel_matrix(N * N);
    
    // Generate a random matrix and make it symmetric positive definite
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i <= j) {
                if (i == j) {
                    // Diagonal elements - ensure positive definiteness
                    kernel_matrix[i * N + j] = dis(gen) + 0.5f;
                } else {
                    // Off-diagonal elements
                    float val = dis(gen) * 0.5f;  // Smaller off-diagonal values
                    kernel_matrix[i * N + j] = val;
                    kernel_matrix[j * N + i] = val;  // Symmetric
                }
            }
        }
    }
    
    return kernel_matrix;
}

// Function to print usage information
void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] <kernel_file.cl>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -t <tokens>     Total number of tokens (default: 4, uses predefined 4x4 matrix)" << std::endl;
    std::cout << "  -s <percentage> Percentage of tokens to select (default: 75, range: 1-100)" << std::endl;
    std::cout << "  -h              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " dpp_kernel.cl                    # Use default 4x4 matrix, select 3 tokens" << std::endl;
    std::cout << "  " << program_name << " -t 100 -s 20 dpp_kernel.cl       # Generate 100x100 matrix, select 20 tokens" << std::endl;
    std::cout << "  " << program_name << " -t 50 -s 50 dpp_kernel.cl        # Generate 50x50 matrix, select 25 tokens" << std::endl;
}

int main(int argc, char** argv) {
    try {
        // Default parameters
        int N = 4;  // Default: use predefined 4x4 matrix
        float select_percentage = 75.0f;  // Default: select 75% of tokens
        std::string kernel_file = "dpp_kernel.cl";
        bool use_custom_matrix = false;
        
        // Parse command line arguments
        int opt;
        while ((opt = getopt(argc, argv, "t:s:h")) != -1) {
            switch (opt) {
                case 't':
                    N = std::atoi(optarg);
                    if (N <= 0) {
                        std::cerr << "Error: Number of tokens must be positive" << std::endl;
                        return 1;
                    }
                    use_custom_matrix = true;
                    break;
                case 's':
                    select_percentage = std::atof(optarg);
                    if (select_percentage <= 0 || select_percentage > 100) {
                        std::cerr << "Error: Selection percentage must be between 1 and 100" << std::endl;
                        return 1;
                    }
                    break;
                case 'h':
                    printUsage(argv[0]);
                    return 0;
                default:
                    printUsage(argv[0]);
                    return 1;
            }
        }
        
        // Get kernel file name from remaining arguments
        if (optind < argc) {
            kernel_file = argv[optind];
        }
        
        // Calculate number of tokens to select
        int T = std::max(1, static_cast<int>(std::round(N * select_percentage / 100.0f)));
        
        std::cout << "=== DPP OpenCL Test ===" << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  Total tokens (N): " << N << std::endl;
        std::cout << "  Tokens to select (T): " << T << " (" << select_percentage << "%)" << std::endl;
        std::cout << "  Kernel file: " << kernel_file << std::endl;
        
        // Initialize OpenCL
        DPPOpenCL dpp_ocl;
        
        std::cout << "\n=== Loading kernel file ===" << std::endl;
        dpp_ocl.loadKernel(kernel_file);

        // Test simple kernel first
        std::cout << "\n=== Test Simple Kernel ===" << std::endl;
        dpp_ocl.testSimpleKernel();
        
        // Test DPP algorithm
        std::cout << "\n=== Testing DPP Algorithm ===" << std::endl;
        
        const int batch_size = 1;
        std::vector<float> kernel_data;
        
        if (use_custom_matrix) {
            // Generate custom kernel matrix
            std::cout << "Generating " << N << "x" << N << " kernel matrix..." << std::endl;
            kernel_data = generateKernelMatrix(N);
        } else {
            // Use predefined 4x4 matrix
            kernel_data = {
                // Batch 0
                0.8f, 0.3f, 0.1f, 0.2f,  // token 0 row
                0.3f, 0.9f, 0.4f, 0.1f,  // token 1 row
                0.1f, 0.4f, 0.7f, 0.5f,  // token 2 row
                0.2f, 0.1f, 0.5f, 0.6f   // token 3 row
            };
        }
        
        // Print kernel matrix (only for small matrices)
        if (N <= 10) {
            std::cout << "Input kernel matrix [" << batch_size << "x" << N << "x" << N << "]:" << std::endl;
            for (int i = 0; i < N; i++) {
                std::cout << "  [";
                for (int j = 0; j < N; j++) {
                    std::cout << std::fixed << std::setprecision(3) << kernel_data[i * N + j];
                    if (j < N-1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
        } else {
            std::cout << "Using generated " << N << "x" << N << " kernel matrix (too large to display)" << std::endl;
        }
        
        // Run OpenCL implementation
        auto ocl_result = dpp_ocl.selectTokens(kernel_data, batch_size, N, T);
        
        // Print results
        std::cout << "\nOpenCL DPP Results:" << std::endl;
        for (int b = 0; b < batch_size; b++) {
            std::cout << "Batch " << b << ": [";
            size_t display_count = std::min(ocl_result[b].size(), static_cast<size_t>(10));
            for (size_t i = 0; i < display_count; i++) {
                std::cout << ocl_result[b][i];
                if (i < display_count-1) std::cout << ", ";
            }
            if (ocl_result[b].size() > 10) {
                std::cout << ", +" << (ocl_result[b].size() - 10) << " more";
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
