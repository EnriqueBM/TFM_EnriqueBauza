#include <CL/sycl.hpp>
#include <oneapi/dpl/random>
#include <typeinfo>
#include <rand_funcs.hpp>
#include <tree_rep.hpp>

using namespace tr;

//Heterogeneous random number genration
void rand_vec_int(sycl::queue queue, std::vector<int> &x, int max, int min, const int &seed){
    {
        sycl::buffer<int, 1> x_buf(x.data(), sycl::range<1>(x.size()));

        queue.submit([&] (sycl::handler &cgh) {

            auto x_acc =
            x_buf.template get_access<sycl::access::mode::write>(cgh);

            //cgh.parallel_for<class count_kernel>(sycl::range<1>(x.size()),
            cgh.parallel_for(sycl::range<1>(x.size()),
                [=](sycl::item<1> idx) {
                std::uint64_t offset = idx.get_linear_id();

                // Create minstd_rand engine
                oneapi::dpl::minstd_rand engine(seed, offset);

                // Create int uniform_real_distribution distribution
		oneapi::dpl::uniform_int_distribution<int> distr (min, max);

                // Generate float random number
                auto res = distr(engine);

                // Store results to x_acc
                x_acc[idx] = res;
            });
        });
    }
}

void rand_vec_float(sycl::queue queue, std::vector<float> &x, float max, float min, const int &seed){
    {
        sycl::buffer<float, 1> x_buf(x.data(), sycl::range<1>(x.size()));

        queue.submit([&] (sycl::handler &cgh) {

            auto x_acc =
            x_buf.template get_access<sycl::access::mode::write>(cgh);

            //cgh.parallel_for<class count_kernel>(sycl::range<1>(x.size()),
            cgh.parallel_for(sycl::range<1>(x.size()),
                [=](sycl::item<1> idx) {
                std::uint64_t offset = idx.get_linear_id();

                // Create minstd_rand engine
                oneapi::dpl::minstd_rand engine(seed, offset);

                // Create float uniform_real_distribution distribution
                oneapi::dpl::uniform_real_distribution<float> distr (min, max);

                // Generate float random number
                auto res = distr(engine);

                // Store results to x_acc
                x_acc[idx] = res;
            });
        });
    }
}
    
rnd::Rand_gen::Rand_gen(int _seed, sycl::queue _q, std::vector<float> _v_f_1, std::vector<float> _v_f_3, std::vector<int> _v_t, std::vector<int> _v_u, std::vector<int> _v_n, std::vector<int> _v_b, std::vector<int> _v_l,
	std::vector<int> _v_args, std::vector<int> _v_s){
		q =_q;
		seed = _seed;
		v_f_1 = _v_f_1; i_f_1 = 0;
		v_f_3 = _v_f_3; i_f_3 = 0;
		v_t = _v_t; i_t = 0;
		v_u = _v_u; i_u = 0;
		v_n = _v_n; i_n = 0;
		v_b = _v_b; i_b = 0;
		v_l = _v_l; i_l = 0;
		v_args = _v_args; i_args = 0;
		v_s = _v_s; i_s = 0;
}
void rnd::Rand_gen::init(){
	rand_vec_float(q, v_f_3, 3.0, -3.0, seed);
	rand_vec_float(q, v_f_1, 1.0, 0.0, seed);
	rand_vec_int(q, v_t, 4, 2, seed);
	rand_vec_int(q, v_u, tr::n_u_fn-1, 0, seed);
	rand_vec_int(q, v_n, tr::n_n_fn-1, 0, seed);
	rand_vec_int(q, v_b, tr::n_b_fn-1, 0, seed);
	rand_vec_int(q, v_l, 1 , 0, seed);
	rand_vec_int(q, v_args, tr::n_args-1, 0, seed);
	rand_vec_int(q, v_s, tr::n_symbols-1, 0, seed);
}


float rnd::Rand_gen::get_rand_float() {

   float value = v_f_1[i_f_1];
   i_f_1 = i_f_1 == v_f_1.size() -1 ? 0 : i_f_1 + 1;
   //std::cout<<i_f_1<<"\n";
   return value;
}

float rnd::Rand_gen::get_rand_const() {

   float value = v_f_3[i_f_3];
   i_f_3 = i_f_3 == v_f_3.size() -1 ? 0 : i_f_3 + 1;
   //std::cout<<i_f_3<<"\n";
   return value;
}


int rnd::Rand_gen::get_rand_type() {

   int value = v_t[i_t];
   i_t = i_t == v_t.size() -1 ? 0 : i_t + 1;
   //std::cout<<i_t<<"\n";
   return value;
}

int rnd::Rand_gen::get_rand_ufn() {

   int value = v_u[i_u];
   i_u = i_u == v_u.size() -1 ? 0 : i_u + 1;
   //std::cout<<i_u<<"\n";
   return value;
}

int rnd::Rand_gen::get_rand_nfn() {

   int value = v_n[i_n];
   i_n = i_n == v_n.size() -1 ? 0 : i_n + 1;
   //std::cout<<i_n<<"\n";
   return value;
}

int rnd::Rand_gen::get_rand_bfn() {

   int value = v_b[i_b];
   i_b = i_b == v_b.size() -1 ? 0 : i_b + 1;
   //std::cout<<i_b<<"\n";
   return value;
}

int rnd::Rand_gen::get_rand_leaf() {

   int value = v_l[i_l];
   i_l = i_l == v_l.size() -1 ? 0 : i_l + 1;
   //std::cout<<i_l<<"\n";
   return value;
}

int rnd::Rand_gen::get_rand_arg() {

   int value = v_args[i_args];
   i_args = i_args == v_args.size() -1 ? 0 : i_args + 1;
   //std::cout<<i_args<<"\n";
   return value;
}

int rnd::Rand_gen::get_rand_symb() {

   int value = v_s[i_s];
   i_s = i_s == v_s.size() -1 ? 0 : i_s + 1;
   //std::cout<<i_s<<"\n";
   return value;
}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
 






