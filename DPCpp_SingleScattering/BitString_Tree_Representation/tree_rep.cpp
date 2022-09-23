#include <tree_rep.hpp>
#include <rand_funcs.hpp>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <cmath>        // std::abs
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <functional>
#include "oneapi/tbb.h"





using namespace std;
using namespace tr;
using namespace rnd;



//Validation functions

void tr::ev_obj(vector<float> &samps, const std::vector<std::array<float, n_symbols>> &points, std::function<float(float)> func){
    for (size_t i=0; i<points.size(); i++){
        samps[i] = func(points[i][0]);
    }
}

std::array<float, n_symbols> gen_point(const std::array<float, n_symbols> &ranges, Rand_gen &rg){
    std::array<float, n_symbols> p;
    for(size_t i=0; i<n_symbols; i++){
    	if(i==2){
    		p[i] = (rg.get_rand_float() - 0.5) * ranges[i]; //tl between -10 and 10
	} else{
        	p[i] = rg.get_rand_float() * ranges[i];
        }
    }

    return p;
}

float tr::rand_float(float a, float b){ // ok
    return a + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(b-a)));
}

int tr::count_args(const term &t){ // ok
    switch(get_type(t)){
        case 0: //float
            return 0;
        case 1: //symbol
            return 0;
        case 2: //unary
            return 1;
        case 3://nary
        {
            int cnt = 0;
            for (size_t i = 0; i < t.d.n_fn.args.size(); i++){
                if (t.d.n_fn.args[i]){
                    cnt += 1;
                }
            }
            return cnt;
        }
        case 4: //binary
            return 2;

    }
}

std::string type_to_string(uint8_t type) { //ok

    static const std::array<std::string, 4> strings{"float", "symbol", "fn", "n-ary"};

    return strings[type];
}

std::string ufn_to_string(uint8_t ufn) { //ok

    static const std::array<std::string, n_u_fn+1> strings{"sin", "exp", "cos"};

    return strings[ufn];
}

std::string nfn_to_string(uint8_t nfn) { //ok

    static const std::array<std::string, 2> strings{"*", "+"};

    return strings[nfn];
}

std::string bfn_to_string(uint8_t bfn) { //ok

    static const std::array<std::string, 1> strings{"/"};

    return strings[bfn];
}

std::string symbol_to_string(uint8_t s) { //ok

    static const std::array<std::string, n_symbols> strings{"t", "r"};

    return strings[s];
}

void tr::print_term (const term& t){ //ok
    if (get_type(t) == 0x0){ // term is a float value
        cout<< t.d.c.value;
    } else if (get_type(t) == 0x1){ // term is a symbol
        cout<< symbol_to_string(get_symbol(t));

    } else if (get_type(t) == 0x2){ //term is a unary function
        if (t.d.u_fn.arg_pt){
            cout<< ufn_to_string(t.d.u_fn.func)<< "(";
            print_term(*t.d.u_fn.arg_pt); // print the argument
            cout<< ")";
        } /*else {
            cout<< ufn_to_string(t.d.u_fn.func)<< "(";
            cout<< "null";
            cout<< ")";
        }*/
    } else if (get_type(t) == 0x3){ // term is a n-ary function
        //int cnt = count_args(t);
        cout<< "(" ;
        for (int i = 0; i < n_args; i++){
            if (t.d.n_fn.args[i]){
                print_term(*t.d.n_fn.args[i]);
            }
            if (t.d.n_fn.args[i+1] && i < (n_args-1)){
            	cout<< " " << nfn_to_string(t.d.n_fn.func)<<" ";
            }
             /*else {
                cout<< "null";
                if (i != (t.d.n_fn.args.size()-1)){
                    cout<< " " << nfn_to_string(t.d.n_fn.func)<<" ";
                }
                //cout<< "null";
            }*/
        }
        cout<< ")" ;
    } else if (get_type(t) == 0x4){ // term is a binary function
        if (t.d.b_fn.args[0] && t.d.b_fn.args[1]){
            cout<< "(" ;
            print_term(*t.d.b_fn.args[0]);
            cout<< " " << bfn_to_string(t.d.b_fn.func);
            print_term(*t.d.b_fn.args[1]);
            cout<< ")" ;
        } /*else {
            cout<< "(" ;
            cout<< "null";
            cout<< " " << bfn_to_string(t.d.b_fn.func);
            cout<< "null";
            cout<< ")" ;
        }*/

    }
}

float tr::evaluate(const term& t,  const std::array<float, n_symbols>& point){ //ok
    switch(get_type(t)){
        case 0x0: //foat
            return get_float_val(t);

        case 0x1: //symbol
            return point[get_symbol(t)]; //symbol acts as index

        case 0x2: //unary func
            switch(get_ufn(t)){
                case 0x0: //sin
                    return sin(evaluate(*t.d.u_fn.arg_pt, point)); //recursive, evaluate the arguments
                case 0x1: //exp
                    return exp(evaluate(*t.d.u_fn.arg_pt, point));
                case 0x2: //cos
                    return cos(evaluate(*t.d.u_fn.arg_pt, point));
            }
        case 0x3: //n_ary func
            switch(get_nfn(t)){
                case 0x0: // product
                {
                    //int cnt = count_args(t);
                    float product = 0.0;
                    int cnt = 0;
                    for(size_t i = 0; i < n_args; i++){
                        if(t.d.n_fn.args[i] && cnt == 0){ //evalua primer argumento no nulo
                           cnt++;
                           product = evaluate(*t.d.n_fn.args[i], point); //recursive, evaluate the arguments
                        } else if(t.d.n_fn.args[i] && cnt != 0){ //evalua resto de argumentos no nulos
                           product *= evaluate(*t.d.n_fn.args[i], point);
                        }
                    }
                    return product;
                }
                case 0x1: // sum
                {
                    //int cnt = count_args(t);
                    float sum = 0.0;
                    for(size_t i = 0; i < n_args; i++){
                        if(t.d.n_fn.args[i]){
                            sum += evaluate(*t.d.n_fn.args[i], point); //recursive, evaluate the arguments
                        }
                    }
                    return sum;
                }
            }
        case 0x4: // binary
            switch(get_bfn(t)){
                case 0x0: //div
                {
                    float div = 0.0;
		     if(get_type(*t.d.b_fn.args[0]) == symb && get_type(*t.d.b_fn.args[1]) == symb && get_symbol(*t.d.b_fn.args[0]) == get_symbol(*t.d.b_fn.args[1])){
     				div = 1.0;
		     }else{
		            float num = evaluate(*t.d.b_fn.args[0], point);
		            float den = evaluate(*t.d.b_fn.args[1], point);
		            div = num / den;
                    }

                    return div;
                }
            }
    }
}

term gen_leaf(){ //ok
    term leaf;
    int type_id = rand() % 2; //generate a float or symbol
    set_type(leaf, types[type_id]);
    if (get_type(leaf) == float_val){ //term is a unary function
        float value = rand_float(-3.0, 3.0);
        set_float_val(leaf, value);
    } else if (get_type(leaf) == symb){
        int symbol_id = rand() % n_symbols; //generate a symbol
        set_symbol(leaf, symbols[symbol_id]);
    }

    return leaf;
}

term tr::gen_branch(const int depth){ //ok
    term t;
    if (depth == 1){
        t = gen_leaf();
        return t;
    }else{
        int n_depth = depth - 1;
        int type_id = rand() % 3 + 2; //generate a n unary, n_ary or binary func
        //int type_id = 3; //n_ary func
        //int type_id = 2; //unary func
        set_type(t, types[type_id]);
        if (get_type(t) == unary){ //term is a unary function
            int u_func_id = rand() % n_u_fn;
            set_u_func(t, u_funcs[u_func_id]); // randomly select a unary func
            //term arg = gen_branch(depth-1); //generate random argument recursively
            term* arg = new term;
            *arg = gen_branch(n_depth);
            t.d.u_fn.arg_pt = arg;
            //set_u_arg(t, arg);

            return t;
        } else if (get_type(t) == nary){
            int n_func_id = rand() % n_n_fn;
            set_n_func(t, n_funcs[n_func_id]); // randomly select a nary func
            for (size_t i = 0; i < n_args; i++){ // randomly generate arguments
                term* arg = new term;
                *arg = gen_branch(n_depth);
                t.d.n_fn.args[i] = arg;
            }

            return t;
        } else if (get_type(t) == binary){
            int b_func_id = rand() % 1;
            set_b_func(t, b_funcs[b_func_id]); // randomly select a binary func
            for (size_t i = 0; i < 2; i++){ // randomly generate arguments
                term* arg = new term;
                *arg = gen_branch(n_depth);
                t.d.b_fn.args[i] = arg;
            }

            return t;
        }
    }
}

term gen_leaf_GPU(Rand_gen &rg){ //ok
    term leaf;
    //int type_id = rand() % 2; //generate a float or symbol
    int type_id = rg.get_rand_leaf();
    set_type(leaf, types[type_id]);
    if (get_type(leaf) == float_val){ //term is a unary function
        //float value = rand_float(-3.0, 3.0);
        float value = rg.get_rand_const();
        set_float_val(leaf, value);
    } else if (get_type(leaf) == symb){
        //int symbol_id = rand() % n_symbols; //generate a symbol
        int symbol_id = rg.get_rand_symb();
        set_symbol(leaf, symbols[symbol_id]);
    }

    return leaf;
}

term gen_branch_GPU(const int depth, Rand_gen &rg){ //ok
    term t;
    if (depth == 1){
        t = gen_leaf_GPU(rg);
        return t;
    }else{
        int n_depth = depth - 1;
        int type_id = rg.get_rand_type();
        set_type(t, types[type_id]);
        if (get_type(t) == unary){ //term is a unary function
            //int u_func_id = rand() % n_u_fn;
            int u_func_id = rg.get_rand_ufn();
            set_u_func(t, u_funcs[u_func_id]); // randomly select a unary func
            //term arg = gen_branch(depth-1); //generate random argument recursively
            term* arg = new term;
            *arg = gen_branch_GPU(n_depth, rg);
            t.d.u_fn.arg_pt = arg;
            //set_u_arg(t, arg);

            return t;
        } else if (get_type(t) == nary){
            int n_func_id = rg.get_rand_nfn();
            set_n_func(t, n_funcs[n_func_id]); // randomly select a nary func
            for (size_t i = 0; i < n_args; i++){ // randomly generate arguments
                term* arg = new term;
                *arg = gen_branch_GPU(n_depth, rg);
                t.d.n_fn.args[i] = arg;
            }

            return t;
        } else if (get_type(t) == binary){
            //int b_func_id = rand() % 1;
            int b_func_id = rg.get_rand_bfn();
            set_b_func(t, b_funcs[b_func_id]); // randomly select a binary func
            for (size_t i = 0; i < 2; i++){ // randomly generate arguments
                term* arg = new term;
                *arg = gen_branch_GPU(n_depth, rg);
                t.d.b_fn.args[i] = arg;
            }

            return t;
        }
    }
}

std::array<term, n_inds> tr::gen_pop(const int depth){ //ok
    std::array<term, n_inds> pop;
    for (size_t i=0; i<n_inds; i++){
        //pop[i] = int_ss_num();
        pop[i] = gen_branch(rand()%depth+1);
        /*if (rand_float(0.0, 1.0) < 0.05){
            pop[i] = int_ss_num();
        } else{
            pop[i] = gen_branch(rand()%depth+1);
        }*/

    }

    return pop;
}

std::array<term, n_inds> tr::gen_pop_GPU(const int depth, Rand_gen &rg){ //ok
    std::array<term, n_inds> pop;
    for (size_t i=0; i<n_inds; i++){
        //pop[i] = int_ss_num();
        pop[i] = gen_branch_GPU(rand()%depth+1, rg);
        /*if (rand_float(0.0, 1.0) < 0.05){
            pop[i] = int_ss_num();
        } else{
            pop[i] = gen_branch(rand()%depth+1);
        }*/

    }

    return pop;
}

bool is_leaf(const term t){ //ok
    if (get_type(t) == float_val || get_type(t) == symb){
        return true;
    }
    return false;

}

term tr::tmnt_selection(std::array<term, n_inds> pop, std::array<float, n_inds> scores){ //ok
    int idx1, idx2, idx3;
    //term* cand = new term;
    term first;
    term second;
    term third;
    float r = rand_float(0.0, 1.0);

    idx1 = rand() % n_inds;
    do{
        idx2 = rand() % n_inds;
    }
    while (idx1 == idx2);
    do{
        idx3 = rand() % n_inds;
    }while(idx3 == idx1 || idx3 == idx2);

    if (scores[idx1] < scores[idx2] && scores[idx1] < scores[idx3]){
        first = pop[idx1];
        //copy_term(pop[idx1], first);
        if (scores[idx2] < scores[idx3]){
            second = pop[idx2]; third = pop[idx3];
            //copy_term(pop[idx2], second); copy_term(pop[idx3], third);
        } else{
            second = pop[idx3]; third = pop[idx2];
            //copy_term(pop[idx3], second); copy_term(pop[idx2], third);
        }

    } else if (scores[idx2] < scores[idx1] && scores[idx2] < scores[idx3]){
        first = pop[idx2];
        //copy_term(pop[idx2], first);
        if (scores[idx1] < scores[idx3]){
            second = pop[idx1]; third = pop[idx3];
            //copy_term(pop[idx1], second); copy_term(pop[idx3], third);
        } else{
            second = pop[idx3]; third = pop[idx1];
            //copy_term(pop[idx3], second); copy_term(pop[idx1], third);
        }
    } else {
        first = pop[idx3];
        //copy_term(pop[idx3], first);
        if (scores[idx1] < scores[idx2]){
            second = pop[idx1]; third = pop[idx2];
            //copy_term(pop[idx1], second); copy_term(pop[idx2], third);
        } else{
            second = pop[idx2]; third = pop[idx1];
            //copy_term(pop[idx2], second); copy_term(pop[idx1], third);
        }
    }

    if (r < tmnt_rates[2]){
        return third;
        /*copy_term(third, *cand);
        free_term(third); free_term(second); free_term(first);
        return cand;*/
    } else if (r < tmnt_rates[1] + tmnt_rates[2]){
        return second;
        /*copy_term(second, *cand);
        free_term(third); free_term(second); free_term(first);
        return cand;*/
    } else {
        return first;
        /*copy_term(first, *cand);
        free_term(third); free_term(second); free_term(first);
        return cand;*/
    }
}

term tmnt_selection_GPU(std::array<term, n_inds> pop, std::array<float, n_inds> scores, Rand_gen &rg){ //ok
    int idx1, idx2, idx3;
    //term* cand = new term;
    term first;
    term second;
    term third;
    //float r = rand_float(0.0, 1.0);
    float r = rg.get_rand_float();

    idx1 = rand() % n_inds;
    do{
        idx2 = rand() % n_inds;
    }
    while (idx1 == idx2);
    do{
        idx3 = rand() % n_inds;
    }while(idx3 == idx1 || idx3 == idx2);

    if (scores[idx1] < scores[idx2] && scores[idx1] < scores[idx3]){
        first = pop[idx1];
        //copy_term(pop[idx1], first);
        if (scores[idx2] < scores[idx3]){
            second = pop[idx2]; third = pop[idx3];
            //copy_term(pop[idx2], second); copy_term(pop[idx3], third);
        } else{
            second = pop[idx3]; third = pop[idx2];
            //copy_term(pop[idx3], second); copy_term(pop[idx2], third);
        }

    } else if (scores[idx2] < scores[idx1] && scores[idx2] < scores[idx3]){
        first = pop[idx2];
        //copy_term(pop[idx2], first);
        if (scores[idx1] < scores[idx3]){
            second = pop[idx1]; third = pop[idx3];
            //copy_term(pop[idx1], second); copy_term(pop[idx3], third);
        } else{
            second = pop[idx3]; third = pop[idx1];
            //copy_term(pop[idx3], second); copy_term(pop[idx1], third);
        }
    } else {
        first = pop[idx3];
        //copy_term(pop[idx3], first);
        if (scores[idx1] < scores[idx2]){
            second = pop[idx1]; third = pop[idx2];
            //copy_term(pop[idx1], second); copy_term(pop[idx2], third);
        } else{
            second = pop[idx2]; third = pop[idx1];
            //copy_term(pop[idx2], second); copy_term(pop[idx1], third);
        }
    }

    if (r < tmnt_rates[2]){
        return third;
        /*copy_term(third, *cand);
        free_term(third); free_term(second); free_term(first);
        return cand;*/
    } else if (r < tmnt_rates[1] + tmnt_rates[2]){
        return second;
        /*copy_term(second, *cand);
        free_term(third); free_term(second); free_term(first);
        return cand;*/
    } else {
        return first;
        /*copy_term(first, *cand);
        free_term(third); free_term(second); free_term(first);
        return cand;*/
    }
}

void tr::tree_random_branch(term &t, std::vector<term*> &terms){
    switch(get_type(t)){
        case 0x0: //foat
            terms.push_back(&t);
            break;
        case 0x1: //symbol
            terms.push_back(&t);
            break;
        case 0x2: //unary func
            terms.push_back(&t);
            tree_random_branch(*t.d.u_fn.arg_pt, terms);
            break;
        case 0x3: //n_ary func
        {
            terms.push_back(&t);
            int idx = rand() % n_args;
            while(!t.d.n_fn.args[idx]){
                idx = rand() % n_args;
            }
            tree_random_branch(*t.d.n_fn.args[idx], terms);
        }
            break;
        case 0x4: // binary
        {
            terms.push_back(&t);
            int idx = rand() % 2;
            tree_random_branch(*t.d.b_fn.args[idx], terms);
        }
            break;
    }
}

void tree_random_branch_GPU(term &t, std::vector<term*> &terms, Rand_gen &rg){
    switch(get_type(t)){
        case 0x0: //foat
            terms.push_back(&t);
            break;
        case 0x1: //symbol
            terms.push_back(&t);
            break;
        case 0x2: //unary func
            terms.push_back(&t);
            tree_random_branch_GPU(*t.d.u_fn.arg_pt, terms, rg);
            break;
        case 0x3: //n_ary func
        {
            terms.push_back(&t);
            //int idx = rand() % n_args;
            int idx = rg.get_rand_arg();
            while(!t.d.n_fn.args[idx]){
                idx = rand() % n_args;
            }
            tree_random_branch_GPU(*t.d.n_fn.args[idx], terms, rg);
        }
            break;
        case 0x4: // binary
        {
            terms.push_back(&t);
            //int idx = rand() % 2;
            int idx = rg.get_rand_nfn(); //getting a randon int between 0 and 1.
            tree_random_branch_GPU(*t.d.b_fn.args[idx], terms, rg);
        }
            break;
    }
}


term* tr::select_branch(term &t){
    std::vector<term*> branch_terms;
    tree_random_branch(t, branch_terms);
    int idx = rand() % branch_terms.size();
    return branch_terms[idx];
}

term* select_branch_GPU(term &t, Rand_gen &rg){
    std::vector<term*> branch_terms;
    tree_random_branch_GPU(t, branch_terms, rg);
    int idx = rand() % branch_terms.size();
    return branch_terms[idx];
}

term* tr::select_branch_leaf(term &t){
    std::vector<term*> branch_terms;
    tree_random_branch(t, branch_terms);
    int idx;
    if (rand_float(0.0, 1.0) < 0.7){
        idx = branch_terms.size()-1;
    } else {
        idx = rand() % (branch_terms.size() - 1);
    }
    return branch_terms[idx];
}

void tr::mute_1(const term &in, term& out){
    copy_term(in, out);
    mute_1_rec(out);
}

void tr::mute_1_rec(term& t){ //ok
    //Replaces a leaf by a subtree
    int dth = rand() % 2 + 2; // Randomly choose the depth
    //int dth = 2;
    //int a = rand() % 2 + 2;  // Randomly choose the number of arguments
    if (!is_leaf(t)){
        if (get_type(t) == unary){ //term is a unary function
            if (is_leaf(*t.d.u_fn.arg_pt) || rand_float(0.0, 1.0) < subtree_mutation_rate){ // its argument is a leaf or rand_f < subtree_rate --> replace the leaf
                free_arg(t.d.u_fn.arg_pt); // deallocate memory from previous argument
                term* arg = new term;
                *arg = gen_branch(dth); // Generate the subtree to insert
                t.d.u_fn.arg_pt = arg; // set new argument
            } else { //argument is not a leaf --> go deeper into the tree
                mute_1_rec(*t.d.u_fn.arg_pt);
            }

        } else if (get_type(t) == nary){ //term is n-ary func
            int idx = rand() % n_args; // Randomly choose the argument index to change
            while(!t.d.n_fn.args[idx]){
                idx = rand() % n_args;
            }
            if (is_leaf(*t.d.n_fn.args[idx]) || rand_float(0.0, 1.0) < subtree_mutation_rate){ // its argument is a leaf or rand_f < subtree_rate --> replace the leaf
                free_arg(t.d.n_fn.args[idx]); // deallocate memory from previous argument
                term* arg = new term;
                *arg = gen_branch(dth); // Generate the subtree to insert
                t.d.n_fn.args[idx] = arg; // set new argument
            } else { //argument is not a leaf --> go deeper into the tree
                mute_1_rec(*t.d.n_fn.args[idx]);
            }
        } else if (get_type(t) == binary){
            int idx = rand() % 2; // Randomly choose the argument index to change
            if (is_leaf(*t.d.b_fn.args[idx]) || rand_float(0.0, 1.0) < subtree_mutation_rate){ // its argument is a leaf or rand_f < subtree_rate --> replace the leaf
                free_arg(t.d.b_fn.args[idx]); // deallocate memory from previous argument
                term* arg = new term;
                *arg = gen_branch(dth); // Generate the subtree to insert
                t.d.b_fn.args[idx] = arg; // set new argument
            } else { //argument is not a leaf --> go deeper into the tree
                mute_1_rec(*t.d.b_fn.args[idx]);
            }
        }
    } else {
        t = gen_branch(dth);
    }
}



void mute_1_rec_GPU(term& t, Rand_gen &rg){ //ok
    //Replaces a leaf by a subtree
    //int dth = rand() % 2 + 2; // Randomly choose the depth
    int dth = rg.get_rand_nfn() + 2;
    if (!is_leaf(t)){
        if (get_type(t) == unary){ //term is a unary function
            if (is_leaf(*t.d.u_fn.arg_pt) || rg.get_rand_float() < subtree_mutation_rate){ // its argument is a leaf or rand_f < subtree_rate --> replace the leaf
                free_arg(t.d.u_fn.arg_pt); // deallocate memory from previous argument
                term* arg = new term;
                *arg = gen_branch_GPU(dth, rg); // Generate the subtree to insert
                t.d.u_fn.arg_pt = arg; // set new argument
            } else { //argument is not a leaf --> go deeper into the tree
                mute_1_rec_GPU(*t.d.u_fn.arg_pt, rg);
            }

        } else if (get_type(t) == nary){ //term is n-ary func
            //int idx = rand() % n_args; // Randomly choose the argument index to change
            int idx = rg.get_rand_arg();
            while(!t.d.n_fn.args[idx]){
                idx = rg.get_rand_arg();
            }
            if (is_leaf(*t.d.n_fn.args[idx]) || rg.get_rand_float() < subtree_mutation_rate){ // its argument is a leaf or rand_f < subtree_rate --> replace the leaf
                free_arg(t.d.n_fn.args[idx]); // deallocate memory from previous argument
                term* arg = new term;
                *arg = gen_branch_GPU(dth, rg); // Generate the subtree to insert
                t.d.n_fn.args[idx] = arg; // set new argument
            } else { //argument is not a leaf --> go deeper into the tree
                mute_1_rec_GPU(*t.d.n_fn.args[idx], rg);
            }
        } else if (get_type(t) == binary){
            //int idx = rand() % 2; // Randomly choose the argument index to change
            int idx = rg.get_rand_nfn();
            if (is_leaf(*t.d.b_fn.args[idx]) || rg.get_rand_float() < subtree_mutation_rate){ // its argument is a leaf or rand_f < subtree_rate --> replace the leaf
                free_arg(t.d.b_fn.args[idx]); // deallocate memory from previous argument
                term* arg = new term;
                *arg = gen_branch_GPU(dth, rg); // Generate the subtree to insert
                t.d.b_fn.args[idx] = arg; // set new argument
            } else { //argument is not a leaf --> go deeper into the tree
                mute_1_rec_GPU(*t.d.b_fn.args[idx], rg);
            }
        }
    } else {
        t = gen_branch_GPU(dth, rg);
    }
}

void mute_1_GPU(const term &in, term& out, Rand_gen &rg){
    copy_term(in, out);
    mute_1_rec_GPU(out, rg);
}

/*void tr::mute_2_rec(term& t, int depth){ //ok
// Reduces the depth of the tree
	free_term(t);
	t = gen_leaf();
}*/

void tr::mute_2(const term &in, term& out){
    copy_term(in, out);
    term* node_to_red = select_branch(out);
    free_term(*node_to_red);
    *node_to_red = gen_leaf();
}

void mute_2_GPU(const term &in, term& out, Rand_gen &rg){
    copy_term(in, out);
    term* node_to_red = select_branch_GPU(out, rg);
    free_term(*node_to_red);
    *node_to_red = gen_leaf_GPU(rg);
}

void tr::mute_5(const term &in, term& out){
    std::vector<term*> vt;
    copy_term(in, out);
    tree_random_branch(out, vt);
    if(vt.size() > 1){
	    int idx_low = rand() % (vt.size()-1) + 1;
	    int idx_high = rand() % idx_low;
	    term* low_node = new term;
	    copy_term(*vt[idx_low], *low_node);
	    free_term(*vt[idx_high]);
	    copy_term(*low_node, *vt[idx_high]);
	    free_arg(low_node);
    }
}

void mute_5_GPU(const term &in, term& out, Rand_gen &rg){
    std::vector<term*> vt;
    copy_term(in, out);
    tree_random_branch_GPU(out, vt, rg);
    if(vt.size() > 1){
	    int idx_low = rand() % (vt.size()-1) + 1;
	    int idx_high = rand() % idx_low;
	    term* low_node = new term;
	    copy_term(*vt[idx_low], *low_node);
	    free_term(*vt[idx_high]);
	    copy_term(*low_node, *vt[idx_high]);
	    free_arg(low_node);
    }
}

void tr::mute_3(const term &in, term& out){
    copy_term(in, out);
    term* node_to_rep = select_branch(out);
    mute_3_rec(*node_to_rep);
}

void tr::mute_3_rec(term& t){ //ok
    //Replaces a node by other node. Maintain the arguments.
    if (is_leaf(t)){ // If node is a leaf replace it by other leaf
        t = gen_leaf();
    } else if(get_type(t) == unary){ // Term is unary function
        int type_id = rand() % 3 + 2; //generate an unary, n_ary or binary func
        if (types[type_id] == unary){ // New function is also unary
            //std::cout<<"Replacing a unary node by unary one "<<std::endl;
            int u_func_id = rand() % n_u_fn; // Randomly select an unary func to replace the old one
            set_u_func(t, u_funcs[u_func_id]); // Replace the node, the argument is the same
        } else if (types[type_id] == nary){ // New func is n-ary
            //std::cout<<"Replacing a unary node by nary one "<<std::endl;
            term* arg_0 = new term;
            copy_term(*t.d.u_fn.arg_pt, *arg_0); // First arguemnt is the same
            free_term(t);
            set_type(t, types[type_id]); // Change the term type to n-ary func
            int n_func_id = rand() % n_n_fn; //Randomly select a n-ary func to replace the old one
            set_n_func(t, n_funcs[n_func_id]); // Replace the node
            t.d.n_fn.args[0] = arg_0;
            for (size_t i = 1; i < n_args; i++){ // randomly the rest of the arguments
                term* arg = new term;
                *arg = gen_branch(rand() % 2 + 1);
                t.d.n_fn.args[i] = arg;
            }
        } else if (types[type_id] == binary){
            //std::cout<<"Replacing a unary node by binary one "<<std::endl;
            term* arg_0 = new term;
            //*arg_0 = *t.d.u_fn.arg_pt; // First arguemnt is the same
            copy_term(*t.d.u_fn.arg_pt, *arg_0); // First arguemnt is the same
            term* arg_1 = new term;
            *arg_1 = gen_branch(rand() % 2 + 1);
            free_term(t);
            set_type(t, types[type_id]); // Change the term type to binary func
            int b_func_id = rand() % 1; //Randomly select a n-ary func to replace the old one
            set_b_func(t, b_funcs[b_func_id]); // Replace the node
            t.d.b_fn.args[0] = arg_0;
            t.d.b_fn.args[1] = arg_1;
        }
    } else if (get_type(t) == nary){
        //int cnt = count_args(t);
        //int idx0 = -1;
        int idx0 = rand() % n_args;
        while(!t.d.n_fn.args[idx0]){
            idx0 = rand() % n_args;
            /*print_term(t);
            std::cout<<"\n";*/
        }
        int type_id = rand() % 3 + 2; //generate an unary, n_ary or binary func
        if (types[type_id] == unary){
            //std::cout<<"Replacing a nary node by unary one "<<std::endl;
            term* arg = new term;
            //*arg = *t.d.n_fn.args[idx0];
            copy_term(*t.d.n_fn.args[idx0], *arg);
            free_term(t); //deallocate meomory
            set_type(t, types[type_id]);
            int u_func_id = rand() % n_u_fn; // Randomly select an unary func to replace the old one
            set_u_func(t, u_funcs[u_func_id]); // Replace the node, the argument is the same
            t.d.u_fn.arg_pt = arg;

        } else if (types[type_id] == nary){ // New func is n-ary
            //std::cout<<"Replacing a nary node by nary one "<<std::endl;
            int n_func_id = rand() % n_n_fn; //Randomly select a n-ary func to replace the old one
            set_n_func(t, n_funcs[n_func_id]); // Replace the node, arguments are the same


        } else if (types[type_id] == binary){
            //std::cout<<"Replacing a nary node by binary one "<<std::endl;
            int b_func_id = rand() % 1; //Randomly select a binary func to replace the old one
            term* arg_0 = new term;
            term* arg_1 = new term;
            int idx1 = rand() % n_args;
            while(!t.d.n_fn.args[idx1] /*|| idx1==idx0*/){
                idx1 = rand() % n_args;
                /*print_term(t);
                std::cout<<"\n";*/
            }
            copy_term(*t.d.n_fn.args[idx0], *arg_0);
            copy_term(*t.d.n_fn.args[idx1], *arg_1);
            free_term(t); //deallocate previous args
            set_type(t, types[type_id]); // Change the term type to binary func
            set_b_func(t, b_funcs[b_func_id]); // Replace the node
            t.d.b_fn.args[0] = arg_0;
            t.d.b_fn.args[1] = arg_1;
        }
    } else if (get_type(t) == binary){
        int idx = rand() % 2;
        int type_id = rand() % 3 + 2; //generate an unary or n_ary func
        if (types[type_id] == unary){ // New func is unary
            //std::cout<<"Replacing a binary node by unary one "<<std::endl;
            term* arg = new term;
            copy_term(*t.d.b_fn.args[idx], *arg);
            free_term(t);
            set_type(t, types[type_id]);
            int u_func_id = rand() % n_u_fn; // Randomly select an unary func to replace the old one
            set_u_func(t, u_funcs[u_func_id]); // Replace the node, the argument is the same
            t.d.u_fn.arg_pt = arg;
        } else if (types[type_id] == nary){ // New func is n-ary
            //std::cout<<"Replacing a binary node by nary one "<<std::endl;
            int n_func_id = rand() % n_n_fn; //Randomly select a n-ary func to replace the old one
            term* arg_0 = new term;
            copy_term(*t.d.b_fn.args[0], *arg_0);
            term* arg_1 = new term;
            copy_term(*t.d.b_fn.args[1], *arg_1);
            free_term(t);
            set_type(t, types[type_id]);
            set_n_func(t, n_funcs[n_func_id]); // Replace the node
            t.d.n_fn.args[0] = arg_0;
            t.d.n_fn.args[1] = arg_1;
            for (size_t i = 2; i < n_args; i++){ // randomly the rest of the arguments
                term* arg = new term;
                *arg = gen_branch(rand() % 2 + 1);
                t.d.n_fn.args[i] = arg;
            }
        } else if (types[type_id] == binary){
            //std::cout<<"Replacing a binary node by binary one "<<std::endl;
            //Same type and arguments, only change binary func type
            int b_func_id = rand() % 1; //Randomly select a binary func to replace the old one
            set_b_func(t, b_funcs[b_func_id]); // Replace the node
        }
    }
}

void mute_3_rec_GPU(term& t, Rand_gen &rg){ //ok
    //Replaces a node by other node. Maintain the arguments.
    if (is_leaf(t)){ // If node is a leaf replace it by other leaf
        t = gen_leaf_GPU(rg);
    } else if(get_type(t) == unary){ // Term is unary function
        //int type_id = rand() % 3 + 2; //generate an unary, n_ary or binary func
        int type_id = rg.get_rand_type();
        if (types[type_id] == unary){ // New function is also unary
            //std::cout<<"Replacing a unary node by unary one "<<std::endl;
            //int u_func_id = rand() % n_u_fn; // Randomly select an unary func to replace the old one
            int u_func_id = rg.get_rand_ufn();
            set_u_func(t, u_funcs[u_func_id]); // Replace the node, the argument is the same
        } else if (types[type_id] == nary){ // New func is n-ary
            //std::cout<<"Replacing a unary node by nary one "<<std::endl;
            term* arg_0 = new term;
            copy_term(*t.d.u_fn.arg_pt, *arg_0); // First arguemnt is the same
            free_term(t);
            set_type(t, types[type_id]); // Change the term type to n-ary func
            //int n_func_id = rand() % n_n_fn; //Randomly select a n-ary func to replace the old one
            int n_func_id = rg.get_rand_nfn();
            set_n_func(t, n_funcs[n_func_id]); // Replace the node
            t.d.n_fn.args[0] = arg_0;
            for (size_t i = 1; i < n_args; i++){ // randomly the rest of the arguments
                term* arg = new term;
                *arg = gen_branch_GPU(rg.get_rand_nfn() + 1, rg); //uses rand_nfn here to obtain a number between 1 and 2, its not a real nary_function
                t.d.n_fn.args[i] = arg;
            }
        } else if (types[type_id] == binary){
            //std::cout<<"Replacing a unary node by binary one "<<std::endl;
            term* arg_0 = new term;
            //*arg_0 = *t.d.u_fn.arg_pt; // First arguemnt is the same
            copy_term(*t.d.u_fn.arg_pt, *arg_0); // First arguemnt is the same
            term* arg_1 = new term;
            *arg_1 = gen_branch_GPU(rg.get_rand_nfn() + 1, rg);
            free_term(t);
            set_type(t, types[type_id]); // Change the term type to binary func
            //int b_func_id = rand() % 1; //Randomly select a n-ary func to replace the old one
            int b_func_id = rg.get_rand_bfn();
            set_b_func(t, b_funcs[b_func_id]); // Replace the node
            t.d.b_fn.args[0] = arg_0;
            t.d.b_fn.args[1] = arg_1;
        }
    } else if (get_type(t) == nary){
        //int cnt = count_args(t);
        //int idx0 = -1;
        //int idx0 = rand() % n_args;
        int idx0 = rg.get_rand_arg();
        while(!t.d.n_fn.args[idx0]){
            idx0 = rg.get_rand_arg();
            /*print_term(t);
            std::cout<<"\n";*/
        }
        //int type_id = rand() % 3 + 2; //generate an unary, n_ary or binary func
        int type_id = rg.get_rand_type();
        if (types[type_id] == unary){
            //std::cout<<"Replacing a nary node by unary one "<<std::endl;
            term* arg = new term;
            //*arg = *t.d.n_fn.args[idx0];
            copy_term(*t.d.n_fn.args[idx0], *arg);
            free_term(t); //deallocate meomory
            set_type(t, types[type_id]);
            //int u_func_id = rand() % n_u_fn; // Randomly select an unary func to replace the old one
            int u_func_id = rg.get_rand_ufn();
            set_u_func(t, u_funcs[u_func_id]); // Replace the node, the argument is the same
            t.d.u_fn.arg_pt = arg;

        } else if (types[type_id] == nary){ // New func is n-ary
            //std::cout<<"Replacing a nary node by nary one "<<std::endl;
            //int n_func_id = rand() % n_n_fn; //Randomly select a n-ary func to replace the old one
            int n_func_id = rg.get_rand_nfn();
            set_n_func(t, n_funcs[n_func_id]); // Replace the node, arguments are the same


        } else if (types[type_id] == binary){
            //std::cout<<"Replacing a nary node by binary one "<<std::endl;
            //int b_func_id = rand() % 1; //Randomly select a binary func to replace the old one
            int b_func_id = rg.get_rand_bfn();
            term* arg_0 = new term;
            term* arg_1 = new term;
            //int idx1 = rand() % n_args;
            int idx1 = rg.get_rand_arg();
            while(!t.d.n_fn.args[idx1] /*|| idx1==idx0*/){
                idx1 = rg.get_rand_arg();
                /*print_term(t);
                std::cout<<"\n";*/
            }
            copy_term(*t.d.n_fn.args[idx0], *arg_0);
            copy_term(*t.d.n_fn.args[idx1], *arg_1);
            free_term(t); //deallocate previous args
            set_type(t, types[type_id]); // Change the term type to binary func
            set_b_func(t, b_funcs[b_func_id]); // Replace the node
            t.d.b_fn.args[0] = arg_0;
            t.d.b_fn.args[1] = arg_1;
        }
    } else if (get_type(t) == binary){
        //int idx = rand() % 2;
        int idx = rg.get_rand_nfn(); 
        //int type_id = rand() % 3 + 2; //generate an unary or n_ary func
        int type_id = rg.get_rand_type();
        if (types[type_id] == unary){ // New func is unary
            //std::cout<<"Replacing a binary node by unary one "<<std::endl;
            term* arg = new term;
            copy_term(*t.d.b_fn.args[idx], *arg);
            free_term(t);
            set_type(t, types[type_id]);
            //int u_func_id = rand() % n_u_fn; // Randomly select an unary func to replace the old one
            int u_func_id = rg.get_rand_ufn();
            set_u_func(t, u_funcs[u_func_id]); // Replace the node, the argument is the same
            t.d.u_fn.arg_pt = arg;
        } else if (types[type_id] == nary){ // New func is n-ary
            //std::cout<<"Replacing a binary node by nary one "<<std::endl;
            //int n_func_id = rand() % n_n_fn; //Randomly select a n-ary func to replace the old one
            int n_func_id = rg.get_rand_nfn();
            term* arg_0 = new term;
            copy_term(*t.d.b_fn.args[0], *arg_0);
            term* arg_1 = new term;
            copy_term(*t.d.b_fn.args[1], *arg_1);
            free_term(t);
            set_type(t, types[type_id]);
            set_n_func(t, n_funcs[n_func_id]); // Replace the node
            t.d.n_fn.args[0] = arg_0;
            t.d.n_fn.args[1] = arg_1;
            for (size_t i = 2; i < n_args; i++){ // randomly the rest of the arguments
                term* arg = new term;
                *arg = gen_branch_GPU(rg.get_rand_nfn() + 1, rg);
                t.d.n_fn.args[i] = arg;
            }
        } else if (types[type_id] == binary){
            //std::cout<<"Replacing a binary node by binary one "<<std::endl;
            //Same type and arguments, only change binary func type
            //int b_func_id = rand() % 1; //Randomly select a binary func to replace the old one
            int b_func_id = rg.get_rand_bfn();
            set_b_func(t, b_funcs[b_func_id]); // Replace the node
        }
    }
}

void mute_3_GPU(const term &in, term& out, Rand_gen &rg){
    copy_term(in, out);
    term* node_to_rep = select_branch_GPU(out, rg);
    mute_3_rec_GPU(*node_to_rep, rg);
}

void tr::mute_4(const term &in, term& out){
    //Chain mutation
    copy_term(in, out);
    term* node_to_chain = select_branch(out);
    mute_4_rec(*node_to_chain);
}

void tr::mute_4_rec(term& t){ //ok
    term* arg = new term;
    copy_term(t, *arg);
    free_term(t);
    set_type(t, unary);
    int u_fn_id = rand() % n_u_fn;
    set_u_func(t, u_funcs[u_fn_id]);
    t.d.u_fn.arg_pt = arg;
}

void mute_4_rec_GPU(term& t, Rand_gen &rg){ //ok
    term* arg = new term;
    copy_term(t, *arg);
    free_term(t);
    set_type(t, unary);
    //int u_fn_id = rand() % n_u_fn;
    int u_fn_id = rg.get_rand_ufn();
    set_u_func(t, u_funcs[u_fn_id]);
    t.d.u_fn.arg_pt = arg;
}

void mute_4_GPU(const term &in, term& out, Rand_gen &rg){
    //Chain mutation
    copy_term(in, out);
    term* node_to_chain = select_branch_GPU(out, rg);
    mute_4_rec_GPU(*node_to_chain, rg);
}


void tree_size_rec(const term &t, int &n_nodes){
   switch(get_type(t)){
        case 0x0: //foat
            n_nodes += 1;
            break;
        case 0x1: //symbol
            n_nodes += 1;
            break;
        case 0x2: //unary func
            n_nodes += 1;
            tree_size_rec(*t.d.u_fn.arg_pt, n_nodes);
            break;
        case 0x3: //n_ary func
            n_nodes += 1;
            for (int i=0; i<n_args; i++){
                if(t.d.n_fn.args[i]){
                    tree_size_rec(*t.d.n_fn.args[i], n_nodes);
                }
            }
            break;
        case 0x4: // binary
            n_nodes += 1;
            for (int i=0; i<2; i++){
                if(t.d.n_fn.args[i]){
                    tree_size_rec(*t.d.b_fn.args[i], n_nodes);
                }
            }
            break;
    }

}

int tr::tree_size (const term &t){
    int n = 0;
    tree_size_rec(t, n);

    return n;
}

void tr::crossover(const term& in, term &t1, term& t2){ //ok, t1 = out, t2 = donnor
    copy_term(in, t1);
    term* branch_donnor = select_branch(t2);
    term* branch_ind = select_branch(t1);
    free_term(*branch_ind);
    copy_term(*branch_donnor, *branch_ind);
}

void crossover_GPU(const term& in, term &t1, term& t2, Rand_gen &rg){ //ok, t1 = out, t2 = donnor
    copy_term(in, t1);
    term* branch_donnor = select_branch_GPU(t2, rg);
    term* branch_ind = select_branch_GPU(t1, rg);
    free_term(*branch_ind);
    copy_term(*branch_donnor, *branch_ind);
}

void tr::evolution(const std::array<term, n_inds> &pop, std::array<term, n_inds/2> &ev_pop, const std::array<float, n_inds> &scores){
    for (size_t i = 0; i < n_inds/2; i++){
    //std::cout<<"evolving ind n: "<<i;
        term candidate = tmnt_selection(pop, scores);
        float r = rand_float(0.0, 1.0);
        if (r < cross_rate){
            //std::cout<<" crossover"<<std::endl;
            term donnor = tmnt_selection(pop, scores);
            crossover(candidate, ev_pop[i], donnor); // crossover
            //free_arg(donnor);
        } else if (r < cross_rate + hoist_rate){
            //std::cout<<" hoist"<<std::endl;
            mute_2(candidate, ev_pop[i]); // reduce mutation
        } else if (r < cross_rate + hoist_rate + point_rate){
            //std::cout<<" point"<<std::endl;
            mute_3(candidate, ev_pop[i]); // replace mutation
        } else if (r < cross_rate + hoist_rate + point_rate + subtree_rate){
            //std::cout<<" subtree"<<std::endl;
            mute_1(candidate, ev_pop[i]);  //subtree mutation
    	} else if (r < cross_rate + hoist_rate + point_rate + subtree_rate + chain_rate){
            //std::cout<<" chain"<<std::endl;
            mute_4(candidate, ev_pop[i]);  //chain mutation
    	} else if (r < cross_rate + hoist_rate + point_rate + subtree_rate + chain_rate + reduce_rate){
            //std::cout<<" reduce"<<std::endl;
            mute_5(candidate, ev_pop[i]);  //chain mutation
        } else {
            //std::cout<<" unevolved"<<std::endl;
            copy_term(candidate, ev_pop[i]);
        }
    }
}

void tr::evolution_GPU(const std::array<term, n_inds> &pop, std::array<term, n_inds/2> &ev_pop, const std::array<float, n_inds> &scores, Rand_gen &rg){
    for (size_t i = 0; i < n_inds/2; i++){
    //std::cout<<"evolving ind n: "<<i;
        term candidate = tmnt_selection_GPU(pop, scores, rg);
        //float r = rand_float(0.0, 1.0);
        float r = rg.get_rand_float();
        if (r < cross_rate){
            //std::cout<<" crossover"<<std::endl;
            term donnor = tmnt_selection_GPU(pop, scores, rg);
            crossover_GPU(candidate, ev_pop[i], donnor, rg); // crossover
            //free_arg(donnor);
        } else if (r < cross_rate + hoist_rate){
            //std::cout<<" hoist"<<std::endl;
            mute_2_GPU(candidate, ev_pop[i], rg); // reduce mutation
        } else if (r < cross_rate + hoist_rate + point_rate){
            //std::cout<<" point"<<std::endl;
            mute_3_GPU(candidate, ev_pop[i], rg); // replace mutation
        } else if (r < cross_rate + hoist_rate + point_rate + subtree_rate){
            //std::cout<<" subtree"<<std::endl;
            mute_1_GPU(candidate, ev_pop[i], rg);  //subtree mutation
    	} else if (r < cross_rate + hoist_rate + point_rate + subtree_rate + chain_rate){
            //std::cout<<" chain"<<std::endl;
            mute_4_GPU(candidate, ev_pop[i], rg);  //chain mutation
    	} else if (r < cross_rate + hoist_rate + point_rate + subtree_rate + chain_rate + reduce_rate){
            //std::cout<<" reduce"<<std::endl;
            mute_5_GPU(candidate, ev_pop[i], rg);  //chain mutation
        } else {
            //std::cout<<" unevolved"<<std::endl;
            copy_term(candidate, ev_pop[i]);
        }
    }
}

float mae (const std::vector<float>& fit){ //ok
    float error = 0.0;
    for(size_t j = 0; j < fit.size(); j++){
        error += fit[j];
    }
    error = error / (float) fit.size();

    return error;
}

float mae_2 (const std::vector<float>& fit, const float &n){ //ok
    float error = 0.0;
    for(size_t j = 0; j < fit.size(); j++){
        error += fit[j];
    }
    error = error / n;

    return error;
}

std::array<float, n_inds> tr::fitness(std::array<term, n_inds> &pop, std::function<float(std::array<float, n_symbols>)> obj, Rand_gen &rg,  const std::array<float, n_symbols> &mins,
 const std::array<float, n_symbols> &maxs){
  
    std::array<float, n_inds> scores;
    std::array<term, n_inds> derivates;
    std::array<float, n_symbols> p;
    std::array<float, n_symbols> ranges;
    float v_obj;
    int n = 6250000; //n_points
    
    oneapi::tbb::parallel_for(size_t(0), (size_t)n_inds, [&](size_t i){
                    derivates[i] = derivate(pop[i], 0x0);
        });
    for(size_t i=0; i<n_symbols; i++){
    	ranges[i] = std::abs(maxs[i] - mins[i]);
    }
    for(size_t i_1=0; i_1<50; i_1++){
	//p[0] = rg.get_rand_float() * ranges[0]; //random t
	p[0] = mins[0] + (ranges[0]/50.0)*i_1;
    	for(size_t i_2=0; i_2<50; i_2++){
		//p[1] = rg.get_rand_float() * ranges[1]; //random dl (r)
		p[1] = mins[1] + (ranges[1]/50.0)*i_2;
		for(size_t i_3=0; i_3< 50; i_3++){
			//p[2] = (rg.get_rand_float() - 0.5) * ranges[2]; //random tl (x)
			p[2] = mins[2] + (ranges[2]/50.0)*i_3;
			for (size_t i_4=0; i_4<50; i_4++){
				//p[3] = rg.get_rand_float() * ranges[3];
				p[3] = mins[3] + (ranges[3]/50.0)*i_4;
				v_obj = obj(p);
				oneapi::tbb::parallel_for(size_t(0), (size_t)n_inds, [&](size_t i){
				        scores[i] += std::abs(evaluate(derivates[i], p) - v_obj);
                    		});
			} 
		}
    	}
    }
    oneapi::tbb::parallel_for(size_t(0), (size_t)n_inds, [&](size_t k){
	    free_term(derivates[k]);
	    if((!isnan(scores[k])) && (!isinf(scores[k]))){
		scores[k] = scores[k]/n;
	    }else{
		scores[k] = 1000000000.0;
	    }
    });
    
    return scores;
}

std::array<float, n_inds> tr::fitness_2(std::array<term, n_inds> &pop, std::function<float(std::array<float, n_symbols>)> obj, Rand_gen &rg,  const std::array<float, n_symbols> &mins,
 const std::array<float, n_symbols> &maxs){
  
    std::array<float, n_inds> scores;
    std::array<term, n_inds> derivates;
    std::array<float, n_symbols> p;
    std::array<float, n_symbols> ranges;
    size_t n = 200000; //n_points
    mutex scoreMutex;
    
    oneapi::tbb::parallel_for(size_t(0), (size_t)n_inds, [&](size_t i){
                    derivates[i] = derivate(pop[i], 0x0);
    });
    oneapi::tbb::parallel_for(size_t(0), (size_t)n, [&](size_t i){
    	std::array<float, n_symbols> p = gen_point(ranges, rg);
    	float v_obj = obj(p);
    	for(size_t j=0; j<n_inds; j++){
		scoreMutex.lock();
		scores[j] += std::abs(evaluate(derivates[j], p) - v_obj);
		scoreMutex.unlock();
    	}
    	
    });
    oneapi::tbb::parallel_for(size_t(0), (size_t)n_inds, [&](size_t k){
	    free_term(derivates[k]);
	    if((!isnan(scores[k])) && (!isinf(scores[k]))){
		scores[k] = scores[k]/n;
	    }else{
		scores[k] = 10000000000.0;
	    }
    });
    
    return scores;
}

std::array<float, n_inds> tr::fitness_ss(std::array<term, n_inds> &pop, std::function<float(std::array<float, n_symbols>)> obj, const int &n_p, const std::array<float, n_symbols> &mins,
    const std::array<float, n_symbols> &maxs){
        std::array<float, n_inds> scores = {0};
        std::array<int, n_symbols> idxs = {0};
        //std::array<int, n_inds> sizes;
        std::array<term, n_inds> derivates;
        std::array<int, n_symbols> steps;
        for(size_t i=0; i<n_symbols; i++){
            steps[i] = std::abs(maxs[i] - mins[i])/dim_samps[i];
        }
        //int max_size = 0;
        int n = 0;
        //COmpute max size for size penalization
        /*for (size_t i = 0; i < n_inds; i++){
            int n_nodes = tree_size(pop[i]);
            sizes[i] = n_nodes;
            if (n_nodes > max_size){
                max_size = n_nodes;
            }
        }*/
        //Generate derivates
        oneapi::tbb::parallel_for(size_t(0), (size_t)n_inds, [&](size_t i){
                    derivates[i] = derivate(pop[i], 0x0);
        });

        std::array<float, n_symbols> p; //Generate random point
        for (int j=0; j<n_p; j++){
            //p = gen_point(mins, maxs);

            for (size_t d=0; d<n_symbols; d++){
                p[d] = mins[d] + steps[d] * idxs[d];
                float v_obj = obj(p);
                //float v_obj = 1/((p[0]-p[2])*(p[0]-p[2]) + p[1]*p[1]);
                //If random point is in obj domain evaluate individuals
                if((!isnan(v_obj)) && (!isinf(v_obj))){
                    n++;

                    oneapi::tbb::parallel_for(size_t(0), (size_t)n_inds, [&](size_t i){
                        /*term d_ind;
                        d_ind = derivate(pop[i], 0x0);*/
                        scores[i] += std::abs(evaluate(derivates[i], p) - v_obj);
                        //scores[i] = std::abs(evaluate(pop[i], p) - v_obj);
                        //free_term(d_ind);
                    });

                }
            }
            idxs[0]++;
            for (size_t d=1; d<n_symbols; d++){ //reinicia todos los indices menos el ultimo dependiendo del anterior
                if(idxs[d-1] == dim_samps[d-1]){
                    idxs[d-1] = 0;
                    idxs[d]++;
                }
            }
            if (idxs[n_symbols] == dim_samps[n_symbols]){ //reinicia el ultimo indice
                idxs[n_symbols] = 0;
            }



        }
        //Apply mae and size penalization
        oneapi::tbb::parallel_for(size_t(0), (size_t)n_inds, [&](size_t k){
                    free_term(derivates[k]);
                    if((!isnan(scores[k])) && (!isinf(scores[k]))){
                        scores[k] = scores[k]/n;
                        /*float size_const = pow((float)sizes[k]/max_size, 6.0) + 1.0;
                        scores[k] = scores[k] * size_const;*/
                    }else{
                        scores[k] = 100000000000.0;
                    }
        });
        return scores;
}

std::array<float, n_inds> tr::fitness_ss2(std::array<term, n_inds> &pop, std::function<float(std::array<float, n_symbols>)> obj, const std::array<std::array<float, n_symbols>, n_samps> &samps,
    const float &i_obj){
        std::array<float, n_inds> scores = {0};
        //std::array<int, n_inds> sizes;
        //std::array<term, n_inds> derivates;
        //int max_size = 0;
        //int n = 0;

        //Compute max size for size penalization
        /*for (size_t i = 0; i < n_inds; i++){
            int n_nodes = tree_size(pop[i]);
            sizes[i] = n_nodes;
            if (n_nodes > max_size){
                max_size = n_nodes;
            }
        }*/

        for (size_t i = 0; i < n_inds; i++){
        //oneapi::tbb::parallel_for(size_t(0), (size_t)n_inds, [&](size_t i){
            //std::vector<float> fit (n_samps);
            term d_ind = derivate(pop[i], 0x0);
            /*print_term(pop[i]);
            std::cout<<"\n";
            print_term(d_ind);
            std::cout<<"\n\n";*/
            for(size_t j = 0; j < n_samps; j++){
                float v_obj = obj(samps[j]);
                //if((!isnan(v_obj)) && (!isinf(v_obj))){
                    //fit[j] = std::abs(evaluate(d_ind, samps[j]) - v_obj);
                    scores[i] += std::abs(evaluate(d_ind, samps[j]) - v_obj);
                    //scores[i] += std::abs(evaluate(pop[i], samps[j]) - v_obj);
                    //n++;
                /*}else{
                    fit[j] = 0.0;
                }*/

            }
            /*oneapi::tbb::parallel_for(size_t(0), (size_t)points.size(), [&](size_t j){
                float v_obj = obj(samps[j]);
                if((!isnan(v_obj)) && (!isinf(v_obj))){
                    fit[j] = std::abs(evaluate(d_ind, samps[j]) - v_obj);
                    //fit[j] = std::abs(evaluate(pop[i], samps[j]) - v_obj);
                    n++;
                }else{
                    fit[j] = 0.0;
                }
                }
            );*/

            //float size_const = pow((float)sizes[i]/max_size, 6.0) + 1.0;
            //float error = mae_2(fit, i_obj) /** size_const*/;

            //if ((!isnan(error)) && (!isinf(error))){
            if ((!isnan(scores[i])) && (!isinf(scores[i]))){
                scores[i] = (scores[i]/i_obj)/* * size_const */;
            } else {
                scores[i] = 100000000000.0;
            }
            free_term(d_ind);
        }
        //); //parallel_for end
    return scores;
}

bool tr::depends_on(const term& t, uint8_t var){ //ok
    switch(get_type(t)){
        case 0x0: //float
            return false;
        case 0x1: //symbol
            if (get_symbol(t) == var){
                return true;
            } else {
                return false;
            }
        case 0x2: //unary
            return depends_on(*t.d.u_fn.arg_pt, var);
        case 0x3: //n_ary
        {
            bool aux = false;
            //int cnt = count_args(t);
            for (size_t i = 0; i < t.d.n_fn.args.size(); i++){
                if(t.d.n_fn.args[i]){
                    aux = aux || depends_on(*t.d.n_fn.args[i], var);
                }
            }
            return aux;
        }
        case 0x4: //binary
        {
            bool aux = false;
            for (size_t i = 0; i < 2; i++){
                aux = aux || depends_on(*t.d.b_fn.args[i], var);
            }
            return aux;
        }
    }
}

term tr::derivate(const term& t, uint8_t var){ //ok
    term der;
    if (!depends_on(t, var)){
            set_type(der, float_val);
            set_float_val(der, 0.0);

            return der;
    } else {
        switch(get_type(t)){
            case 0: //float
                set_type(der, float_val);
                set_float_val(der, 0.0);

                return der;
            case 1: //symbol
                set_type(der, float_val);
                if (get_symbol(t) == var){
                    set_float_val(der, 1.0);

                    return der;
                } else {
                    set_float_val(der, 0.0);

                    return der;
                }
            case 2: //unary func, apply "regla de la cadena"
                switch(get_ufn(t)){
                    case 0x0: //sin(f(x)) = cos(f(x))*f'(x)
                    {
                        set_type(der, nary);
                        set_n_func(der, n_funcs[0]); // new expression is a product

                        term* cos_arg = new term;
                        copy_term(*t.d.u_fn.arg_pt, *cos_arg);

                        term* cos = new term;
                        set_type(*cos, unary);
                        set_u_func(*cos, u_funcs[2]); //cosine
                        cos->d.u_fn.arg_pt = cos_arg;

                        term* arg_2 = new term;
                        *arg_2 = derivate(*t.d.u_fn.arg_pt, var);

                        der.d.n_fn.args[0] = cos; // first product argument is the cosine
                        der.d.n_fn.args[1] = arg_2; // second product argument is the derivate of the argument

                        return der;
                    }
                    case 0x1: //exp(f(x)) = exp(f(x))*f'(x)
                    {
                        set_type(der, nary);
                        set_n_func(der, n_funcs[0]); // new expression is a product

                        term* arg = new term;
                        copy_term(t, *arg); // first product argument is the same exponential

                        term* arg2 = new term;
                        *arg2 = derivate(*t.d.u_fn.arg_pt, var); // second product argument is the derivate of the argument

                        der.d.n_fn.args[0] = arg;
                        der.d.n_fn.args[1] = arg2; 

                        return der;
                    }
                }
            case 3: //n_ary func
                switch(get_nfn(t)){
                    case 0x0: // product; (f(x)*g(x)*h(x))' = f'(x)*g(x)*h(x) + f(x)*g'(X)*h(x) + f(x)*g(x)*h'(x)
                    {
                        set_type(der, nary);
                        set_n_func(der, n_funcs[1]); // new expression is a sum


                        for(size_t i = 0; i < n_args; i++){
                            if (t.d.n_fn.args[i] && depends_on(*t.d.n_fn.args[i], var)){
                                term* arg = new term;
                                set_type(*arg, nary);
                                set_n_func(*arg, n_funcs[0]); // each summand is a product

                                for(size_t j = 0; j < n_args; j++){
                                    if (i == j){
                                        term* arg_j = new term;
                                        *arg_j = derivate(*t.d.n_fn.args[j], var);
                                        arg->d.n_fn.args[j] = arg_j;
                                    } else {
                                        if (t.d.n_fn.args[j]){
                                            term* arg_j = new term;
                                            copy_term(*t.d.n_fn.args[j], *arg_j);
                                            arg->d.n_fn.args[j] = arg_j;
                                        }
                                    }
                                }
                                der.d.n_fn.args[i] = arg;
                            }
                        }
                        return der;

                    }
                    case 0x1: // sum; (f(x) + g(x))' = f'(x) + g'(x)
                    {
                        set_type(der, nary);
                        set_n_func(der, n_funcs[1]); // new expression is a sum


                        for (size_t i = 0; i < n_args; i++){
                            if (t.d.n_fn.args[i] && depends_on(*t.d.n_fn.args[i], var)){
                                term* arg = new term;
                                *arg = derivate(*t.d.n_fn.args[i], var); 
                                der.d.n_fn.args[i] = arg;
                            }
                        }

                        return der;
                    }
                }
            case 4: //binary
                switch(get_bfn(t)){
                    case 0x0: // div
                    {
                        //(n(x)/d(x))' = (n'(x)*d(x) - n(x)*d'(x)) / d²(x)
                        set_type(der, binary);
                        set_b_func(der, b_funcs[0]); // new expression is a div
                        term* num = new term;
                        term* den = new term;

                        set_type(*den, nary); //denom is a product
                        set_n_func(*den, n_funcs[0]); // den = d(x) * d(x)
                        term* den_1 = new term;
                        term* den_2 = new term;
                        copy_term(*t.d.b_fn.args[1], *den_1);
                        copy_term(*t.d.b_fn.args[1], *den_2);
                        den->d.n_fn.args[0] = den_1;
                        den->d.n_fn.args[1] = den_2;

                        set_type(*num, nary); // num is a sum
                        set_n_func(*num, n_funcs[1]); //num = n'(x)*d(x) - n(x)*d'(x)

                        term* num_1 = new term;
                        set_type(*num_1, nary); // num_1 is a product
                        set_n_func(*num_1, n_funcs[0]); // num_1 = n'(x)*d(x)
                        term* num_1_1 = new term; // f'(x)
                        term* num_1_2 = new term; // g(x)
                        *num_1_1 = derivate(*t.d.b_fn.args[0], var);
                        copy_term(*t.d.b_fn.args[1], *num_1_2);

                        num_1->d.n_fn.args[0] = num_1_1;
                        num_1->d.n_fn.args[1] = num_1_2;

                        term* num_2 = new term;
                        set_type(*num_2, nary);
                        set_n_func(*num_2, n_funcs[0]); // num_2 = (-1)*n(x)*d'(x)

                        term* one = new term;
                        set_type(*one, float_val);
                        set_float_val(*one, -1.0);


                        term* num_2_2 = new term; //num_2_2 = f(x)*g'(x)
                        set_type(*num_2_2, nary);
                        set_n_func(*num_2_2, n_funcs[0]);


                        term* num_2_2_1 = new term;
                        copy_term(*t.d.b_fn.args[0], *num_2_2_1);
                        term* num_2_2_2 = new term;
                        *num_2_2_2 = derivate(*t.d.b_fn.args[1], var);

                        num_2_2->d.n_fn.args[0] = num_2_2_1;
                        num_2_2->d.n_fn.args[1] = num_2_2_2;

                        num_2->d.n_fn.args[0] = one;
                        num_2->d.n_fn.args[1] = num_2_2;

                        num->d.n_fn.args[0] = num_1;
                        num->d.n_fn.args[1] = num_2;

                        der.d.b_fn.args[0] = num;
                        der.d.b_fn.args[1] = den;

                        return der;
                    }
                }
        }
    }
}

std::array<std::array<float, n_symbols>, n_samps> tr::gen_samps (std::array<float, n_symbols> minimum, std::array<float, n_symbols> maximum){ //ok
    std::array<std::array<float, n_symbols>, n_samps> samps;
    for (size_t i=0; i<n_symbols; i++){
        float step = (maximum[i] - minimum[i]) / n_samps;
        for (size_t j=0; j<n_samps; j++){
            samps[j][i] = minimum[i] + step*j;
        }

    }
    return samps;
}

std::array<std::array<float, n_symbols>, n_samps> tr::gen_samps_2 (std::array<float, n_symbols> minimum, std::array<float, n_symbols> maximum){ //ok
    std::array<std::array<float, n_symbols>, n_samps> samps;
    std::array<int, n_symbols> idxs;
    std::array<float, n_symbols> steps;
    for (size_t i=0; i<n_symbols; i++){
        idxs[i] = 0;
    }
    for(size_t i=0; i<n_symbols; i++){
        steps[i] = std::abs(maximum[i] - minimum[i])/dim_samps[i];
    }
    std::cout<<" "<<std::endl;
    for (size_t cnt=0; cnt<n_samps; cnt++){
        for (size_t d=0; d<n_symbols; d++){
            samps[cnt][d] = minimum[d] + steps[d] * idxs[d];

        }
        idxs[0]++;
        for (size_t d=1; d<n_symbols; d++){ //reinicia todos los indices menos el ultimo dependiendo del anterior
            if(idxs[d-1] == dim_samps[d-1]){
                idxs[d-1] = 0;
                idxs[d]++;
            }
        }
        if (idxs[n_symbols] == dim_samps[n_symbols]){ //reinicia el ultimo indice
            idxs[n_symbols] = 0;
        }
    }
    return samps;
}

void tr::copy_term (const term& in, term &out){
    set_type(out, get_type(in));
    switch(get_type(in)){
            case 0: //float
                set_float_val(out, get_float_val(in));
                break;
            case 1: //symbol
                set_symbol(out, get_symbol(in));
                break;
            case 2: //unary func
            {
                term* out_arg = new term;
                set_u_func(out, get_ufn(in));
                copy_term(*in.d.u_fn.arg_pt, *out_arg);
                out.d.u_fn.arg_pt = out_arg;

            }
            break;
            case 3: //n_ary func
            {
                set_n_func(out, get_nfn(in));
                for (size_t i=0; i<n_args; i++){
                    if(in.d.n_fn.args[i]){
                        term* arg_i = new term;
                        copy_term(*in.d.n_fn.args[i], *arg_i);
                        out.d.n_fn.args[i] = arg_i;
                    }
                }
            }
            break;
            case 4: //binary
            {
                set_b_func(out, get_bfn(in));
                for (size_t i=0; i<2; i++){
                    if(in.d.b_fn.args[i]){
                        term* arg_i = new term;
                        copy_term(*in.d.b_fn.args[i], *arg_i);
                        out.d.b_fn.args[i] = arg_i;
                    }
                }
            }
            break;
        }
 }

void tr::free_term(term &t){
    switch(get_type(t)){
            case 0: //float
                break;
            case 1: //symbol
                break;
            case 2: //unary func
                free_arg(t.d.u_fn.arg_pt);
                //t.d.u_fn.arg_pt = nullptr;
                break;
            case 3: //n_ary func
                for (size_t i=0; i<t.d.n_fn.args.size(); i++){
                    if(t.d.n_fn.args[i]){
                        free_arg(t.d.n_fn.args[i]);
                        //t.d.n_fn.args[i] = nullptr;
                    }
                }
                break;
            case 4: //binary
                for (size_t i=0; i<t.d.b_fn.args.size(); i++){
                    if(t.d.b_fn.args[i]){
                        free_arg(t.d.b_fn.args[i]);
                        //t.d.b_fn.args[i] = nullptr;
                    }
                }
                break;
        }
}

void tr::free_arg(term* &arg){
    if(arg){
        switch(get_type(*arg)){
            case 0: //float
                break;
            case 1: //symbol
                break;
            case 2: //unary func
                free_arg(arg->d.u_fn.arg_pt);
                break;
            case 3: //n_ary func
                for (size_t i=0; i<arg->d.n_fn.args.size(); i++){
                    if(arg->d.n_fn.args[i]){
                        free_arg(arg->d.n_fn.args[i]);
                    }
                }
                break;
            case 4: //binary
                for (size_t i=0; i<arg->d.b_fn.args.size(); i++){
                    if(arg->d.b_fn.args[i]){
                        free_arg(arg->d.b_fn.args[i]);
                    }
                }
                break;
        }
        delete arg;
        arg = nullptr;
    }
}

void tr::free_pop(std::array<term, n_inds> &pop){
    for(size_t i=0; i<pop.size(); i++){
        free_term(pop[i]);
    }
}

void tr::free_ev_pop(std::array<term, n_inds/2> &pop){
    for(size_t i=0; i<pop.size(); i++){
        free_term(pop[i]);
    }
}

void tr::copy_pop(const std::array<term, n_inds> &pop_in, std::array<term, n_inds> &pop_out){
    for(size_t i=0; i<pop_in.size(); i++){
        copy_term(pop_in[i], pop_out[i]);
    }
}

void tr::write_pop(std::ofstream &file_name, std::array<term, n_inds> &pop){
    for(size_t i=0; i<n_inds; i++){
        write_term(pop[i], file_name);
    }
}

void write_term_rec(const term &t, std::ofstream &file_name){
    switch(get_type(t)){
            case 0: //float
                file_name<<0<<" "<<t.d.c.value<<" ";
                break;
            case 1: //symbol
                file_name<<1<<" "<<(int)t.d.s.var<<" ";
                break;
            case 2: //unary func
                file_name<<2<<" "<<(int)t.d.u_fn.func<<" ";
                write_term_rec(*t.d.u_fn.arg_pt, file_name);
                break;
            case 3: //n_ary func
                file_name<<3<<" "<<(int)t.d.n_fn.func<<" "<<count_args(t)<<" ";
                for (size_t i=0; i<t.d.n_fn.args.size(); i++){
                    if(t.d.n_fn.args[i]){
                        write_term_rec(*t.d.n_fn.args[i], file_name);
                    }
                }
                break;
            case 4: //binary
                file_name<<4<<" "<<(int)t.d.b_fn.func<<" ";
                for (size_t i=0; i<t.d.b_fn.args.size(); i++){
                    if(t.d.b_fn.args[i]){
                        write_term_rec(*t.d.b_fn.args[i], file_name);
                    }
                }
                break;
    }
}

void tr::write_term(const term &t, std::ofstream &file_name){
    write_term_rec(t, file_name);
    file_name<<'\n';
}

term tr::read_file (std::ifstream &inFile){
    term ind;
    if(!inFile){
        std::cout<<"Unable to open file";
        exit(1);

    }
    std::string str;
    getline(inFile, str, '\n');
    std::cout<<str<<std::endl;
    std::vector<std::string> out;
    std::stringstream ss(str);
    std::string s;
    int i = 0;

    while (std::getline(ss, s, ' ')){
        out.push_back(s);
        //std::cout<< s << '\n';
    }
    read_term(out, i, ind);

    return ind;

}

void tr::read_term (const std::vector<std::string> &t_vector, int &id, term &out){
    switch(std::stoi(t_vector[id])){
        case 0: //float
            set_type(out, types[0]);
            set_float_val(out, std::stof(t_vector[id+1]));
            id = id + 2;
            break;
        case 1: //symbol
            set_type(out, types[1]);
            set_symbol(out, symbols[std::stoi(t_vector[id+1])]);
            id = id + 2;
            break;
        case 2: //unary function
        {
            set_type(out, types[2]);
            set_u_func(out, u_funcs[std::stoi(t_vector[id+1])]);
            id = id + 2;
            term* arg = new term;
            read_term(t_vector, id, *arg);
            out.d.u_fn.arg_pt = arg;
        }
        break;
        case 3: //n_ary func
        {
            set_type(out, types[3]);
            set_n_func(out, n_funcs[std::stoi(t_vector[id+1])]);
            int n_args = std::stoi(t_vector[id+2]);
            id = id + 3;
            for (size_t j=0; j<n_args; j++){
                term* arg = new term;
                read_term(t_vector, id, *arg);
                out.d.n_fn.args[j] = arg;
            }
        }
        break;
        case 4: //binary_func
        {
            set_type(out, types[4]);
            set_b_func(out, b_funcs[std::stoi(t_vector[id+1])]);
            id = id + 2;
            for (size_t j=0; j<2; j++){
                term* arg = new term;
                read_term(t_vector, id, *arg);
                out.d.b_fn.args[j] = arg;
            }
        }
    }
}

void tr::join_arrs (const std::array<term, n_inds> &pop, const std::array<term, n_inds/2> &ev_pop, std::array<term, n_inds> &new_pop, const std::array<size_t, n_inds> &sort_idxs){

    for (int i = 0; i < n_inds; i++)
    {
        if (i < n_inds/2) {
            copy_term(pop[sort_idxs[i]], new_pop[i]);
        }
        else {
            copy_term(ev_pop[i - n_inds/2], new_pop[i]);
        }
    }
}



void tr::argsort(const std::array<float, n_inds> &scores, std::array<size_t, n_inds> &indices) {
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&scores](size_t left, size_t right) -> bool {
                  // sort indices according to corresponding array element
                  return scores[left] < scores[right];});
}













