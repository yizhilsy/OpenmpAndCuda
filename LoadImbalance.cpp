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
    #pragma omp parallel for num_threads(4) reduction(+:sum)
    for(int i=0;i<10000000;i++) {
        if(i<5000000) {
            sum += smallWork();
        }else {
            sum += bigWork();
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"Not Balance, The Sum:"<<sum<<",time:"<<elapsed.count()<<endl;
    
    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(4) reduction(+:sum)
    for(int i=0;i<10000000;i++) {
        if(i%2==0) {
            sum += smallWork();
        }else {
            sum += bigWork();
        }
    }
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    cout<<"Balance, The Sum:"<<sum<<",time:"<<elapsed.count()<<endl;

    sum = 0;
    start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<10000000;i++) {
        if(i<5000000) {
            sum += smallWork();
        }else {
            sum += bigWork();
        }
    }
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    cout<<"Serial, The Sum:"<<sum<<",time:"<<elapsed.count()<<endl;

    return 0;
}

int smallWork() {
    int sum = 0;
    for(int i=0;i<10;i++) {
        sum += i;
    }
    return sum;
}
int bigWork() {
    int sum = 0;
    for(int i=0;i<1000;i++) {
        sum += i;
    }
    return sum;
}