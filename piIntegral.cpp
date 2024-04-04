#include <iostream>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <cstring>
#include <chrono>
using namespace std;
typedef long long LL;
const int numSteps = 1000000000;

// 串行算法
void Serial() {
    double step = 1.0 / (double)numSteps;
    double sum = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for(double i = 0; i<numSteps; i++) {
        double x = (i + 0.5) * step;
        double y = 4.0 / (1.0 + x*x);
        sum += step*y;
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"The Pai Integer:"<<sum<<",time:"<<elapsed.count()<<endl;
}

// 使用归约和for法
void ParallelA() {
    double step = 1.0 / (double)numSteps;
    double sum = 0;
    // 定义每个线程计算的行数
    const int OMP_THREAD_NUM = 8;
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 1000000) reduction(+:sum) num_threads(OMP_THREAD_NUM)
    for(int i = 0; i<numSteps; i++) {
        double x = ((double)i + 0.5) * step;
        double y = 4.0 / (1.0 + x*x);
        sum += step*y;
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"Using Reduction and dynamic Schedule, The Pai Integer:"<<sum<<",time:"<<elapsed.count()<<endl;
}

// 使用经典法
void ParallelB() {
    double step = 1.0 / (double)numSteps;
    double sum = 0;double partSum = 0;
    int OMP_THREAD_NUM = 8;
    int LINE_FOR_THREADS = (numSteps + OMP_THREAD_NUM - 1) / OMP_THREAD_NUM;
    if(LINE_FOR_THREADS==0) {
        OMP_THREAD_NUM = numSteps;
        LINE_FOR_THREADS = 1;
    }
    double sumArray[OMP_THREAD_NUM];
    memset(sumArray,0,sizeof(sumArray));
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for private(partSum) num_threads(OMP_THREAD_NUM) 
    for(int i=0; i<OMP_THREAD_NUM; i++) {
        for(int j=i*LINE_FOR_THREADS; j<(i+1)*LINE_FOR_THREADS && j<numSteps; j++) {
            double x = ((double)j+0.5) * step;
            double y = 4.0 / (1.0 + x*x);
            partSum += y*step;
        }
        sumArray[i] = partSum;
    }
    for(int i=0;i<OMP_THREAD_NUM;i++) {
        sum += sumArray[i];
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"Using Parallel Region, "<<"The Pai Integer:"<<sum<<",time:"<<elapsed.count()<<endl;
}

// 使用private和critical法
void ParallelC() {
    double step = 1.0 / (double)numSteps;
    double sum = 0;double x;double pi = 0.0;
    clock_t startTime,endTime;double totalTime;
    // 定义每个线程计算的行数
    const int OMP_THREAD_NUM = 8;
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel private(x,sum) num_threads(OMP_THREAD_NUM) shared(pi)
    {
        int id = omp_get_thread_num();
        for(int i=id;i<numSteps;i+=OMP_THREAD_NUM) {
            x = ((double)i + 0.5) * step;
            double y = 4.0 / (1.0 + x*x);
            sum += step*y;
        }
        #pragma omp critical
        {
            pi += sum;
        }
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"Using critical and private, The Pai Integer:"<<pi<<",time:"<<elapsed.count()<<endl;
}

// 共享任务结构，任务队列法
// 共享任务结构是指多个线程共享同一个任务队列，每个线程从队列中获取任务并执行。
// 这种方式可以提高线程的利用率，因为当一个线程完成了它的任务时，
// 它可以从共享队列中获取更多的任务来执行，而不必等待其他线程。
void ParallelD() {
    double step = 1.0 / (double)numSteps;
    const int OMP_THREAD_NUM = 8;
    const int LINE_FOR_THREADS = (numSteps + OMP_THREAD_NUM - 1) / OMP_THREAD_NUM;
    double pi = 0.0;
    
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel num_threads(OMP_THREAD_NUM)
    {
        #pragma omp single nowait
        {
            for(int i=0;i<OMP_THREAD_NUM;i++) {
                #pragma omp task firstprivate(i)
                {
                    // cout<<omp_get_thread_num()<<endl;
                    double partSum = 0.0;
                    for(int j=i*LINE_FOR_THREADS;j<(i+1)*LINE_FOR_THREADS && j<numSteps;j++) {
                        double x = ((double)j + 0.5)* step;
                        double y =  4.0 / (1.0 + x*x);
                        partSum += y*step;
                    }
                    #pragma omp atomic
                    pi += partSum;
                }
            }
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"Using Task Queue, The Pai Integer:"<<pi<<",time:"<<elapsed.count()<<endl;

}


int main()
{
    Serial();
    ParallelA();
    // ParallelB();
    // ParallelC();
    // ParallelD();
    return 0;
}