#include <iostream>
#include <chrono>
#include <omp.h>
using namespace std;

const int stepNum = 10000;
int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    __int64 sum=0;
    for(int i=0;i<stepNum;i++)
        sum+=i;
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"Using Serial, The Sum:"<<sum<<",time:"<<elapsed.count()<<endl;
    
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<10000;i++)
        sum+=i;
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    cout<<"Using Parallel, The Sum:"<<sum<<",time:"<<elapsed.count()<<endl;
    return 0;
}