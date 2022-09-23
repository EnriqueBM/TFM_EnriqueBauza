#include <iostream>
#include <tree_rep.hpp>
#include <stdlib.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <CL/sycl.hpp>
#include <rand_funcs.hpp>



using namespace tr;
using namespace rnd;

float cosine(const float &x){
    return cos(x);
}

float sq_cos(const float &x){
    return cos(x)*cos(x);
}

float x2inverse(const float &x){
    return 1.0/(x*x);
}

float pol2(const float &x){
    return x*x-((2*x))+3.0;
}

float arctan(const float &x){
    return atan(x);
}

float SingScat_den(const std::array<float, n_symbols> &p){ // t, dl, tl = p[0], p[1], p[2]
    return 1/((p[0]-p[2])*(p[0]-p[2]) + p[1]*p[1]);
}

float SingScat_num(const std::array<float, n_symbols> &p){
    return exp(-p[1]*p[0]);
}

float int_SingScat_num(const std::array<float, n_symbols> &p){
    return exp(-p[1]*p[0])/-p[1];
}

float SingScat(const std::array<float, n_symbols> &p){
    return exp(-0.1*(p[0]+sqrt((p[0]-0.1)*(p[0]-0.1) + p[1]*p[1])))/((p[0]-0.1)*(p[0]-0.1) + p[1]*p[1]);
}

float SingScat_1(const std::array<float, n_symbols> &p){
    return exp(-0.1*(p[0]+sqrt((p[0]-0.1)*(p[0]-0.1) + p[1]*p[1])));
}

float SingScat_2(const std::array<float, n_symbols> &p){
    return ((p[0]-0.1)*(p[0]-0.1) + p[1]*p[1]);
}


int main()
{

    //srand((unsigned)time(NULL));
    srand(5);
    
    
    sycl::queue queue(sycl::cpu_selector{});
    std::vector<float> v_f (10000);
    std::vector<float> v_const (10000);
    std::vector<int> v_t(10000);
    std::vector<int> v_u(100);
    std::vector<int> v_n(100);
    std::vector<int> v_b(100);
    std::vector<int> v_l(100);
    std::vector<int> v_args(1000);
    std::vector<int> v_s(500);
    int seed = 10;
    Rand_gen rand_gen(seed, queue, v_f, v_const, v_t, v_u, v_n, v_b, v_l, v_args, v_s);
    rand_gen.init();
    
    std::array<term, n_inds> pop = gen_pop_GPU(3, rand_gen);
    
    /*for (size_t j=0; j<1000; j++){
    	print_term(pop[j]);
    	std::cout<<"\n";
    }*/
    
    
    // Objective, simulation and sampling




    //int n = 64000;
    /*std::array<float, n_symbols> r_mins {0.0, 0.1};
    std::array<float, n_symbols> r_maxs {7.5, 10.0};*/
    std::array<float, n_symbols> r_mins {0.0, 0.05};
    std::array<float, n_symbols> r_maxs {5.0, 7.5};
    std::array<std::array<float, n_symbols>, n_samps> samps = gen_samps_2(r_mins, r_maxs);
    float int_obj = 0;
    float der_val = 0;
    for(int i=0; i<n_samps; i++){
	int_obj += SingScat(samps[i]);
	//der_val += evaluate(der, samps[i]);
	/*std::cout<<samps[i][0]<<" "<<samps[i][1]<<"\n";
	std::cout<<SingScat(samps[i])<<"\n\n";*/
	
    }
    int_obj = std::abs(int_obj);
    std::cout<<int_obj<<"\n";
    //std::cout<<der_val<<"\n";
    
    // Aux variables
    std::array<term, n_inds> new_pop;
    std::array<term, n_inds/2> ev_pop;
    int n_iter = 120;
    std::array<float, n_inds> scores;
    std::array<size_t, n_inds> sort_idx;
    //std::array<float, n_samps> obj_fit;



    //Stats
    std::ofstream time, conv, pop_file, best_ind, ind_vals, obj_vals, pts;
    time.open("time_stats.txt");
    conv.open("conv_stats.txt");
    best_ind.open("best_inds.txt");

    float best_score;
    int n_save;

    for(int i=0; i<n_iter; i++){

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	
	//Evaluate individuals
	scores = fitness_ss2(pop, SingScat, samps, int_obj);
	
	//Get sorted indxs
	argsort(scores, sort_idx);

	best_score = scores[sort_idx[0]];

	//Display best inds
	//if ((i%(n_iter/20))==0){
	    for(size_t j = 0; j < 3; j++){
	        std::cout<<scores[sort_idx[j]]<<std::endl;
	    }
	    print_term(pop[sort_idx[0]]);
	    std::cout<<"\n\n";
	    term d_ind = derivate(pop[sort_idx[0]], 0x0);
	    print_term(d_ind);
	    std::cout<<"\n";
	    free_term(d_ind);
	    //write_term(pop[sort_idx[0]], best_ind);
	    std::cout<<i<<"\n\n";
	//}
	
	//Stop condition based on error
	if (/*best_score < 0.01 ||*/ i == n_iter-1){
	    std::cout<<best_score<<std::endl;
	    std::cout<<i<<"\n";
	    print_term(pop[sort_idx[0]]);
	    std::cout<<"\n";
	    //best_ind.open("best_inds.txt");
	    write_term(pop[sort_idx[0]], best_ind);
	    //best_ind.close();
	    std::cout<<"\n";
	    //std::cout<<"sale"<<"\n";
	    break;
	}

	//Save pop after n generations
	/*std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	if (std::chrono::duration_cast<std::chrono::seconds> (t2 - begin).count() > 60){
	    n_save = 5;
	}else{
	    n_save = 5;
	}
	if (i%n_save == 0){
	    best_ind.open("best_inds.txt");
	    write_term(pop[sort_idx[0]], best_ind);
	    best_ind.close();
	    std::cout<<"Pop saved in iter: "<<i<<std::endl;
	}*/
	
	if(i%(n_iter/5) == 0) write_term(pop[sort_idx[0]], best_ind);		
	
	

	//Reproduction and replacement
	evolution_GPU(pop, ev_pop, scores, rand_gen);
	join_arrs(pop, ev_pop, new_pop, sort_idx);
	free_pop(pop);
	free_ev_pop(ev_pop);
	std::swap(new_pop, pop);
	free_pop(new_pop);


	//Stats
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	time <<std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count()<<" ";
	conv <<best_score<<" ";
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
	//std::cout<<i<<std::endl;
    }
    time.close();
    conv.close();
    best_ind.close();


    //Save last pop
    pop_file.open("pop.txt");
    write_pop(pop_file, pop);
    pop_file.close();
	  
    return 0;
}
