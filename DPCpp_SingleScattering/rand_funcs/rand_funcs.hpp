#pragma once

#include <stdlib.h>
#include <CL/sycl.hpp>
#include <oneapi/dpl/random>

namespace rnd {
//Hterogeneous implementation functions

class Rand_gen {
  private:
	sycl::queue q;
	std::vector<float> v_f_1;
	std::vector<float> v_f_3;
	std::vector<int> v_t;
	std::vector<int> v_u;
	std::vector<int> v_n;
	std::vector<int> v_b;
	std::vector<int> v_l;
	std::vector<int> v_args;
	std::vector<int> v_s;
	
	size_t i_f_1, i_f_3, i_t, i_u, i_n, i_b, i_l, i_args, i_s;
	int seed;
   public:
    	Rand_gen(int seed, sycl::queue _q, std::vector<float> _v_f_1, std::vector<float> _v_f_3, std::vector<int> _v_t, std::vector<int> _v_u, std::vector<int> _v_n, std::vector<int> _v_b, std::vector<int> _v_l, std::vector<int> _v_args, std::vector<int> _v_s);
    	void init();
    	float get_rand_float();
    	float get_rand_const();
    	int get_rand_type();
    	int get_rand_ufn();
    	int get_rand_nfn();
    	int get_rand_bfn();
    	int get_rand_leaf();
    	int get_rand_arg();
    	int get_rand_symb();
};



} //namespace rnd
