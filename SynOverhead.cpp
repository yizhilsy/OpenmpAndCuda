#include <iostream>
#include <chrono>
#include <omp.h>
using namespace std;

int smallWork();
int bigWork();

int main()
{
    __int64 sum = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<1000000000;i++) {
        sum+=i;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"Serial, The Sum:"<<sum<<",time:"<<elapsed.count()<<endl;
    
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for(int i=0;i<1000000000;i++) {
        #pragma omp critical
        sum+=i;
    }
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    cout<<"critical, The Sum:"<<sum<<",time:"<<elapsed.count()<<endl;

    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<1000000000;i++) {
        sum+=i;
    }
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    cout<<"reduction, The Sum:"<<sum<<",time:"<<elapsed.count()<<endl;

    return 0;
}