#include <iostream>
#include <omp.h>
#include <chrono>
using namespace std;
static long num_steps = 1000000000;
double step;
#define NUM_THREADS 8
int main()
{
    int i;
    double x, pi, sum[NUM_THREADS];
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS); //******
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel private(i,x)                 //******
    {
        int id;
        id = omp_get_thread_num();
        cout<<id<<endl;
        sum[id] = 0; //**
        #pragma omp for      //******
        for (i = 0; i < num_steps; i++)
        {
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"The Pai Integer:"<<pi<<",time:"<<elapsed.count()<<endl;
}