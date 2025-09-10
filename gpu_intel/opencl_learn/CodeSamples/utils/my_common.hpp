#pragma once

#include <cstring>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>

template <typename T>
inline bool is_close(const std::vector<T> &vec1, const std::vector<T> &vec2, const float& thr = 0.0001f)
{
	// 1. Check if the sizes are different
	if (vec1.size() != vec2.size())
	{
		return false;
	}

	// 2. Iterate through elements and compare them
	for (size_t i = 0; i < vec1.size(); ++i)
	{
		if (fabs(vec1[i] - vec2[i]) > thr)
		{
			return false; // Found a differing element
		}
	}

	// If we reach here, sizes are the same and all elements are equal
	return true;
}

template <typename T>
inline void print_diff(const std::vector<T> &vec1, const std::vector<T> &vec2, const float& thr = 0.0001f, bool ret_in_first_diff = false)
{
	// 2. Iterate through elements and compare them
	for (size_t i = 0; i < vec1.size(); ++i)
	{
		if (fabs(vec1[i] - vec2[i]) > thr)
		{
			std::cout << "   vec1[" << i << "] = " << vec1[i] << ", vec2[" << i << "] = " << vec2[i] << ", diff = " << fabs(vec1[i] - vec2[i]) << std::endl;
			if (ret_in_first_diff)
				return;
		}
	}
}

template <typename T>
inline bool is_same(const std::vector<T> &vec1, const std::vector<T> &vec2) {
	return is_close<T>(vec1, vec2, 0);
}


inline std::string get_env_str(std::string str_env)
{
	std::cout << "ENV: " << str_env << ", default: empty."<< std::endl;
	if (std::getenv(str_env.c_str()))
	{
		return std::string(std::getenv(str_env.c_str()));
	}
	return std::string();
}

inline void get_env_bool(const char* str_env, bool& b_out)
{
	std::cout << "ENV: " << str_env << ", default: " << b_out << std::endl;
	char *p8env = std::getenv(str_env);
	if (p8env != nullptr)
	{
		if (std::string(p8env) == std::string("1") ||
			std::string(p8env) == std::string("true") ||
			std::string(p8env) == std::string("True") ||
			std::string(p8env) == std::string("TRUE"))
		{
			b_out = true;
		}
		else
		{
			b_out = false;
		}
	}
}

inline bool get_env_bool(const char *str_env)
{
	bool b_out = false;
	get_env_bool(str_env, b_out);
	return b_out;
}

inline int get_env_int(std::string str_env)
{
	std::cout << "ENV: " << str_env << ", default: -1" << std::endl;
	if (std::getenv(str_env.c_str()))
	{
		return std::atoi(std::getenv(str_env.c_str()));
	}
	return -1;
}

inline void get_env_int(std::string str_env, int& out)
{
	std::cout << "ENV: " << str_env << ", default: " << out << std::endl;
	if (std::getenv(str_env.c_str()))
	{
		out = std::atoi(std::getenv(str_env.c_str()));
	}
}