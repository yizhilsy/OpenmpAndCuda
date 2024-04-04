#include <iostream>
#include <chrono>
#include <omp.h> 
using namespace std;
static long num_steps = 1000000000;
double step; 
#define NUM_THREADS 8
int main () 
{	  
    int i; 	  
    double x, pi, sum = 0.0; 
    step = 1.0/(double) num_steps; 
    omp_set_num_threads(NUM_THREADS);
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum) private(x) 
    for (i=0;i<num_steps; i++){ 
        x = (i+0.5)*step; 
        sum = sum + 4.0/(1.0+x*x);
    } 
    pi = step * sum; 
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"The Pai Integer:"<<pi<<",time:"<<elapsed.count()<<endl;
    // for(int i=0;i<16;i++) {
    //     cout<<repo[i]<<endl;
    // }

    return 0;
} 