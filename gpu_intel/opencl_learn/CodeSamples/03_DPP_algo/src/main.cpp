// Reference:

#include <stdio.h>
#include <iostream>

#include <CL/opencl.hpp>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <chrono>
#include <string>
#include <cstring>

#include "kernel_io.hpp"
#include "dpp_ref.hpp"
#include "mat_diagonal_max_ref.hpp"
#include "my_log.hpp"
#include "my_common.hpp"
#include "my_ocl.hpp"

static size_t g_max_ws_in_one_group[3] = {0};
static cl_uint g_max_compute_units = 0;

// Command line parameters structure
struct CommandLineArgs {
    int batch_size = 1;
    int token_count = (3577+16)/16*16;
    bool one_group = false;  // 默认为false，指定参数时设为true
    bool split_kernel = false;  // 默认为false，指定参数时设为true
    std::string kernel_file;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -b, --batch <num>     Batch size (default: 1)\n";
    std::cout << "  -m, --tokens <num>    Number of tokens (default: " << (3577+16)/16*16 << ")\n";
    std::cout << "  -k, --kernel <file>   OpenCL kernel file path (required)\n";
    std::cout << "  -one_group            Use one group mode (default: false)\n";
    std::cout << "  -split                Use split kernel mode (default: false)\n";
    std::cout << "  -h, --help            Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " -b 2 -m 1792 -split -k kernel.cl\n";
    std::cout << "  " << program_name << " --batch 2 --tokens 1792 --kernel dpp_kernel_split.cl -one_group\n";
}

CommandLineArgs parse_command_line(int argc, char* argv[]) {
    CommandLineArgs args;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch") == 0) {
            if (i + 1 < argc) {
                args.batch_size = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: -b requires a value\n";
                exit(1);
            }
        }
        else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--tokens") == 0) {
            if (i + 1 < argc) {
                args.token_count = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: -m requires a value\n";
                exit(1);
            }
        }
        else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--kernel") == 0) {
            if (i + 1 < argc) {
                args.kernel_file = argv[++i];
            } else {
                std::cerr << "Error: -k requires a kernel file path\n";
                exit(1);
            }
        }
        else if (strcmp(argv[i], "-one_group") == 0) {
            args.one_group = true;  // 指定了参数就设为true，不需要值
        }
        else if (strcmp(argv[i], "-split") == 0) {
            args.split_kernel = true;  // 指定了参数就设为true，不需要值
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        else {
            std::cerr << "Error: Unknown option " << argv[i] << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    return args;
}

std::vector<int> run_dpp_kernel(Tensor &mat, const std::string& kernel_fn, int selected_token_num = 0)
{
	std::string kernel_entry = "dpp_kernel";
	auto my_ocl = CMyTest(kernel_entry, kernel_fn);

	assert(mat.m == mat.n);
	float numerical_threshold = 1e-6f;

	size_t total_tokens_num = mat.m;
	if (selected_token_num == 0) {
		selected_token_num = static_cast<size_t>(total_tokens_num * 0.6);
	}
	std::vector<int> output_ids(selected_token_num, -1);

	int lws_1 = std::min(mat.m, (int)g_max_ws_in_one_group[1]);
	int gws_1 = lws_1;
	// lws_1 = 4;
	// gws_1 = 4;

	// create buffers on the device
	auto context = my_ocl.get_context();
	cl::Buffer buffer_mat(context, CL_MEM_READ_ONLY, sizeof(float) * mat.get_size());
	cl::Buffer buffer_cis(context, CL_MEM_READ_WRITE, sizeof(float) * selected_token_num * total_tokens_num);
	cl::Buffer buffer_di2s(context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens_num);  // diagonal value.
	cl::Buffer buffer_output_ids(context, CL_MEM_READ_WRITE, sizeof(int) * selected_token_num);
	cl::Buffer buffer_best_value(context, CL_MEM_READ_WRITE, sizeof(float));
	cl::Buffer buffer_best_id(context, CL_MEM_READ_WRITE, sizeof(int));

	// write mat to the device
	my_ocl.get_queue()->enqueueWriteBuffer(buffer_mat, CL_TRUE, 0, sizeof(float) * mat.get_size(), mat.data);
	my_ocl.get_queue()->enqueueWriteBuffer(buffer_output_ids, CL_TRUE, 0, sizeof(int) * selected_token_num, output_ids.data());
	
	auto kernel_dpp = my_ocl.get_kernel();
	kernel_dpp.setArg(0, buffer_mat);
	kernel_dpp.setArg(1, buffer_cis);
	kernel_dpp.setArg(2, buffer_di2s);
	kernel_dpp.setArg(3, buffer_output_ids);
	kernel_dpp.setArg(4, mat._b);
	kernel_dpp.setArg(5, mat.m);
	kernel_dpp.setArg(6, selected_token_num);
	kernel_dpp.setArg(7, sizeof(float) * lws_1, nullptr);
	kernel_dpp.setArg(8, sizeof(int) * lws_1, nullptr);
	kernel_dpp.setArg(9, buffer_best_value);
	kernel_dpp.setArg(10, buffer_best_id);
	kernel_dpp.setArg(11, numerical_threshold);

	std::cout << "  == Params:" << std::endl;
	std::cout << "     gws_1 = " << gws_1 << std::endl;
	std::cout << "     lws_1 = " << lws_1 << std::endl;
	std::cout << "     selected_token_num = " << selected_token_num << std::endl;
	std::cout << "     M = " << mat.m << std::endl;

	for (int l = 0; l < 3; l++)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		my_ocl.get_queue()->enqueueNDRangeKernel(kernel_dpp, cl::NullRange, cl::NDRange(mat._b, gws_1, 1), cl::NDRange(mat._b, lws_1, 1));
		my_ocl.get_queue()->finish();
		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << "  == tm = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
	}

	std::cout << "  == Start to copy output from device to host" << std::endl;
	// Copy output from device to host
	my_ocl.get_queue()->enqueueReadBuffer(buffer_output_ids, CL_TRUE, 0, sizeof(int) * selected_token_num, output_ids.data());
	std::cout << "      Copy from device to host finish. size = " << output_ids.size() << std::endl;
	
	// std::sort(output_ids.begin(), output_ids.end());

	// for (auto outp_id : output_ids) {
	// 	std::cout << "  == output_ids = " << outp_id << std::endl;
	// }

	return output_ids;
}

std::vector<int> run_ref(Tensor& mat, int selected_token_num = 0) {
	Config config;
	// Initialize config for testing
	config.visual_tokens_retain_percentage = 60;  // Will keep 3 out of 4 tokens
	config.relevance_weight = 0.5f;
	config.enable_pruning = true;
	config.pruning_debug_mode = false;
	config.use_negative_relevance = false;  // Not using negative correlation as requested
	config.numerical_threshold = 1e-6f;
	config.device = "CPU";
	config.use_ops_model = false;

	auto dpp_selector = std::make_unique<FastGreedyDPP>(config);

	if (selected_token_num == 0) {
		selected_token_num = config.visual_tokens_retain_percentage * mat.m;
	}

	// warm up
	std::vector<std::vector<size_t>> selected_tokens;
	std::cout << "  == calc reference warm up." << std::endl;
	// selected_tokens = dpp_selector->select(mat, selected_token_num);

	auto t1 = std::chrono::high_resolution_clock::now();
	selected_tokens = dpp_selector->select(mat, selected_token_num);
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "  == CPU refer time = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

	std::vector<int> concatenated_vec;
	for (auto st : selected_tokens) {
		for (auto s : st) {
			concatenated_vec.emplace_back(static_cast<int>(s));
		}
	}
	return concatenated_vec;
}

int main(int argc, char* argv[])
{
    // Parse command line arguments
    CommandLineArgs args = parse_command_line(argc, argv);
    
    if (args.kernel_file.empty()) {
        std::cerr << "Error: Kernel file is required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    std::cout << "== Test DPP algorithm. " << std::endl;
    get_device_info(g_max_ws_in_one_group, g_max_compute_units);

    int M = args.token_count;
    int B = args.batch_size;
    bool dpp_one_group = args.one_group;
    bool dpp_spilt_kernel = args.split_kernel;

    // ==================
    std::cout << "== Generate random test data." << std::endl;	
    std::cout << "  M = " << M << std::endl;
    std::cout << "  B = " << B << std::endl;
    std::cout << "  ONE_GROUP = " << (dpp_one_group ? 1 : 0) << std::endl;
    std::cout << "  SPLIT = " << (dpp_spilt_kernel ? 1 : 0) << std::endl;
    std::cout << "  Kernel file = " << args.kernel_file << std::endl;
    
    auto mat = Tensor(B, M, M);
    mat.random_data();
    int selected_token_num = M * 0.5;
    // selected_token_num = 1;
    
    std::cout << "== Start to run DPP Reference." << std::endl;
    std::vector<int> selected_token_ref;
    selected_token_ref = run_ref(mat, selected_token_num);

    std::cout << "== Start to run DPP GPU kernel." << std::endl;
    std::vector<int> selected_token_gpu;
    if (dpp_one_group) {
        selected_token_gpu = run_dpp_kernel(mat, args.kernel_file, selected_token_num);
    }
    else if (dpp_spilt_kernel) {
        selected_token_gpu = run_dpp_split_kernel(mat, g_max_ws_in_one_group, args.kernel_file, selected_token_num);
    }
    else {
        std::cerr << "Error: Must specify either -one_group or -split mode\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "== Ref VS GPU result compare:" << std::endl;
    if (!is_same<int>(selected_token_ref, selected_token_gpu))
    {
        std::cout << "  == Fail, diff as follow:" << std::endl;
        print_diff<int>(selected_token_ref, selected_token_gpu, 0.0001f, true);

        std::cout << "== Failed." << std::endl;
    }
    else
    {
        std::cout << "  == Success." << std::endl;
        std::cout << "== Done." << std::endl;
    }

    return 0;
}