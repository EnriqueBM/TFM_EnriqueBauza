#pragma once

#include <iostream>
#include <array>
#include <vector>
#include <functional>
#include <rand_funcs.hpp>


namespace tr {

constexpr uint8_t float_val = 0;
constexpr uint8_t symb = 1;
constexpr uint8_t unary = 2;
constexpr uint8_t nary = 3;
constexpr uint8_t binary = 4;

// Single Scattering parameters
constexpr int n_u_fn = 2;
constexpr int n_n_fn = 2;
constexpr int n_b_fn = 1;
constexpr int n_args = 4;
constexpr int n_symbols = 2;
constexpr int n_inds = 1000;
constexpr int n_samps = 200000;
constexpr std::array<float, n_symbols> dim_samps {500.0, 400.0};
//constexpr std::array<float, n_symbols> steps {0.1, 2.0, 2.0, 10.0};

constexpr std::array<uint8_t, 5> types {0, 1, 2, 3, 4}; // float, symbol, unary, n-ary, binary
constexpr std::array<uint8_t, n_u_fn+1> u_funcs {0x0, 0x1, 0x2}; // sin, exp, cos
constexpr std::array<uint8_t, n_n_fn> n_funcs {0x0, 0x1}; // mul, sum
constexpr std::array<uint8_t, 1> b_funcs {0x0}; // div
constexpr std::array<uint8_t, n_symbols> symbols {0x0, 0x1}; // t, r


//Genetic programming constants (rates for random process):
constexpr float subtree_mutation_rate = 0.25;
constexpr float reduce_depth_rate = 0.05;
constexpr float node_mutation_prob = 0.5;
constexpr std::array<float, 3> tmnt_rates = {0.8, 0.15, 0.05};
constexpr float cross_rate = 0.4;
constexpr float point_rate = 0.1;
constexpr float hoist_rate = 0.1;
constexpr float reduce_rate = 0.1;
constexpr float subtree_rate = 0.2;
constexpr float chain_rate = 0.1;


//Data structures:
struct Constant {
    float value; //4 bytes
};
struct Symbol {
    uint8_t var; //up to 7 bytes, 2⁷ posible symbols
};

struct term;

struct N_ary_fn {
    std::array<term*, n_args> args;
    uint8_t func; //up to 3 bytes, 2³ posible functions
};

struct Binary_fn {
    std::array<term*, 2> args;
    uint8_t func;
};
struct Unary_fn {
    term* arg_pt;
    uint8_t func; //up to 6 bytes, 2⁶ posible functions
};

union data {
    Constant c;
    Symbol s;
    Unary_fn u_fn;
    N_ary_fn n_fn;
    Binary_fn b_fn;
};

struct term {
    uint8_t type;
    data d;

};

//Term management functions:
template <typename T>
constexpr uint8_t get_type(const T& t){
    return t.type;
}

template <typename T>
constexpr float get_float_val(const T& t){
    return t.d.c.value;
}

template <typename T>
constexpr uint8_t get_symbol(const T& t){
    return t.d.s.var;
}

template <typename T>
constexpr uint8_t get_ufn(const T& t){
    return t.d.u_fn.func;
}

template <typename T>
constexpr uint8_t get_nfn(const T& t){
    return t.d.n_fn.func;
}

template <typename T>
constexpr uint8_t get_bfn(const T& t){
    return t.d.b_fn.func;
}

/*template <typename T>
constexpr void set_type (T& t, uint8_t val){
    t.type = val;
}*/

template <typename T>
constexpr void set_type (T& t, uint8_t val){
    t.type = val;
    if (val==nary){
        for(int i=0; i < n_args; i++){
            t.d.n_fn.args[i] = nullptr;
        }
    } else if (val==binary){
        for(size_t i=0; i < 2; i++){
            t.d.b_fn.args[i] = nullptr;
        }
    } else if (val==unary){
            t.d.u_fn.arg_pt = nullptr;
    }
}

template <typename T>
static void set_float_val (T& t, float val){
    t.d.c.value = val;
}

template <typename T>
static void set_symbol (T& t, uint8_t val){
    t.d.s.var = val;
}

template <typename T>
static void set_n_func (T& t, uint8_t val){
    t.d.n_fn.func = val;
}

template <typename T>
static void set_u_func (T& t, uint8_t val){
    t.d.u_fn.func = val;
}

template <typename T>
static void set_b_func (T& t, uint8_t val){
    t.d.b_fn.func = val;
}

template <typename T>
static void set_u_arg (T& t1, T* t2){
    t1.d.u_fn.arg_pt = t2;
}

/*template <typename T>
static void set_n_arg (T& t1, std::array<T, n_args>& t_args){
    for (size_t i = 0; i < t_args.size(); i++)
        t1.d.n_fn.args[i] = &t_args[i];
}*/

template <typename T>
static void set_n_arg (T& t1, T* t_args, int n=n_args){
    for (size_t i = 0; i < n; i++)
        t1.d.n_fn.args[i] = &t_args[i];
}

template <typename T>
static void set_b_arg (T& t1, T* t_args){
    for (size_t i = 0; i < 2; i++)
        t1.d.b_fn.args[i] = &t_args[i];
}

void print_term (const term& t);

//Population management functions:
float evaluate(const term& t, const std::array<float, n_symbols>& point);
term gen_branch(const int depth);
term  tmnt_selection(std::array<term, n_inds> pop, const std::array<float, n_inds> scores);
//std::array<float, n_inds> fitness(std::array<term, n_inds> &pop, const std::vector<std::array<float, n_symbols>> &points, std::vector<float> &obj_val);
std::array<float, n_inds> fitness(std::array<term, n_inds> &pop, std::function<float(std::array<float, n_symbols>)> obj, rnd::Rand_gen &rg,  const std::array<float, n_symbols> &mins,
 const std::array<float, n_symbols> &maxs);
std::array<float, n_inds> fitness_2(std::array<term, n_inds> &pop, std::function<float(std::array<float, n_symbols>)> obj, rnd::Rand_gen &rg,  const std::array<float, n_symbols> &mins,
 const std::array<float, n_symbols> &maxs);
std::array<float, n_inds> fitness_ss(std::array<term, n_inds> &pop, std::function<float(std::array<float, n_symbols>)> obj, const int &n_p, const std::array<float, n_symbols> &mins,
    const std::array<float, n_symbols> &maxs);
    
std::array<float, n_inds> fitness_ss2(std::array<term, n_inds> &pop, std::function<float(std::array<float, n_symbols>)> obj, const std::array<std::array<float, n_symbols>,
    n_samps> &samps, const float &i_obj);
    
term derivate(const term& t, uint8_t var);
bool depends_on(const term& t, uint8_t var);
float rand_float(float a, float b);
int count_args(const term &t);
std::array<term, n_inds> gen_pop(const int depth);
std::array<std::array<float, n_symbols>, n_samps> gen_samps (std::array<float, n_symbols> minimum, std::array<float, n_symbols> maximum);
std::array<std::array<float, n_symbols>, n_samps> gen_samps_2 (std::array<float, n_symbols> minimum, std::array<float, n_symbols> maximum);
void copy_term (const term& in, term &out);
term objective ();
void free_term(term &t);
void free_arg(term* &arg);
void free_pop(std::array<term, n_inds> &pop);
void copy_pop(const std::array<term, n_inds> &pop_in, std::array<term, n_inds> &pop_out);
void write_term(const term &t, std::ofstream &file_name);
void write_pop(std::ofstream &file_name, std::array<term, n_inds> &pop);
term read_file (std::ifstream &inFile);
void read_term (const std::vector<std::string> &t_vector, int &id, term &out);
void join_arrs (const std::array<term, n_inds> &pop, const std::array<term, n_inds/2> &ev_pop, std::array<term, n_inds> &new_pop, const std::array<size_t, n_inds> &sort_idxs);
void argsort(const std::array<float, n_inds> &array, std::array<size_t, n_inds> &indices);
void free_ev_pop(std::array<term, n_inds/2> &pop);
term* select_branch(term &t);
term* select_branch_leaf(term &t);
void tree_random_branch(term &t, std::vector<term*> &terms);
int tree_size (const term &t);

term int_ss_num();
std::array<term, n_inds> gen_pop_GPU(const int depth, rnd::Rand_gen &rg);
void evolution_GPU(const std::array<term, n_inds> &pop, std::array<term, n_inds/2> &ev_pop, const std::array<float, n_inds> &scores, rnd::Rand_gen &rg);

//Validation function:
//float cosine(const float &x);
void ev_obj(std::vector<float> &samps, const std::vector<std::array<float, n_symbols>> &points, std::function<float(float)> func);

//Genetic operators:
void mute_1(const term &in, term& out);
void mute_1_rec(term& t);
void mute_2(const term &in, term& out);
//void mute_2_rec(term& t);
void mute_2_rec(term& t, int depth=1);
void mute_3(const term &in, term& out);
void mute_3_rec(term& t);
void mute_4(const term &in, term& out);
void mute_4_rec(term& t);
void mute_5(const term &in, term& out);
//void crossover(const term& in, term &t1, const term& t2);
void crossover(const term& in, term &t1, term& t2);
//std::array<term, n_inds> evolution(std::array<term, n_inds> pop, std::array<float, n_inds> scores);
void evolution(const std::array<term, n_inds> &pop, std::array<term, n_inds/2> &new_pop, const std::array<float, n_inds> &scores);
} // namespace tr
