#include <iostream>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include "Matrix.h"
#include <chrono>
using namespace std;

void test01();
void test02();
void experiment();
void Serial();


int main()
{
    // test01();
    // cout<<"**************************"<<endl;
    // test02();
    experiment();
    // Serial();
    return 0;
}

void test01() {
    int r1 = 2,c1 = 2;
    int array1[4]={1,1,1,1};
    int r2 = 2,c2 = 2;
    int array2[4]={2,2,2,3};
    Matrix<int> m1(r1,c1,array1);
    Matrix<int> m2(r2,c2,array2);
    Matrix<int> m3 = m1*m2;
    cout<<m3;
    Matrix<int> m4 = m1.parallelMulti(m2,4);
    cout<<m4;
}

void experiment() {
    std::random_device rd;  // 使用随机设备作为种子
    std::mt19937 gen(rd()); // 使用 Mersenne Twister 引擎
    std::uniform_int_distribution<> dis(1, 10); // 生成范围在1到100之间的均匀分布的整数

    for(int i=1000;i<=3000;i+=1000) {
        cout<<"=========="<<i<<"*"<<i<<"Matrix=========="<<endl;
        int r1=i,c1=i,r2=i,c2=i;
        int arrayA[r1*c1],arrayB[r2*c2];
        for(int j=0;j<r1;j++) {
            for(int k=0;k<c1;k++) {
                int pos = j*c1 + k;
                // 生成随机数
                arrayA[pos] = dis(gen);
                arrayB[pos] = dis(gen);
            }
        }
        Matrix<int> m1(r1,c1,arrayA);
        Matrix<int> m2(r2,c2,arrayB);
        Matrix<int> resm;

        cout<<"*****1 Threads*****"<<endl;
        auto start = std::chrono::high_resolution_clock::now();
        resm = m1.parallelMulti(m2,1);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        cout<<"One Thread Algorithm Time:"<< elapsed.count() << endl;
        cout<<resm.getElem(101,811)<<endl;

        cout<<"*****2 Threads*****"<<endl;
        start = std::chrono::high_resolution_clock::now();
        resm = m1.parallelMulti(m2,2);
        finish = std::chrono::high_resolution_clock::now();
        elapsed = finish - start;
        cout<<"Two Thread Algorithm Time:"<< elapsed.count() <<endl;

        cout<<"*****4 Threads*****"<<endl;
        start = std::chrono::high_resolution_clock::now();
        resm = m1.parallelMulti(m2,4);
        finish = std::chrono::high_resolution_clock::now();
        elapsed = finish - start;
        cout<<"Four Thread Algorithm Time:"<< elapsed.count() <<endl;

        cout<<"*****8 Threads*****"<<endl;
        start = std::chrono::high_resolution_clock::now();
        resm = m1.parallelMulti(m2,8);
        finish = std::chrono::high_resolution_clock::now();
        elapsed = finish - start;
        cout<<"Eight Thread Algorithm Time:"<<elapsed.count()<<endl;

        cout<<"*****16 Threads*****"<<endl;
        start = std::chrono::high_resolution_clock::now();
        resm = m1.parallelMulti(m2,16);
        finish = std::chrono::high_resolution_clock::now();
        elapsed = finish - start;
        cout<<"Sixteen Thread Algorithm Time:"<< elapsed.count() <<endl;

        cout<<"*****32 Threads*****"<<endl;
        start = std::chrono::high_resolution_clock::now();
        resm = m1.parallelMulti(m2,32);
        finish = std::chrono::high_resolution_clock::now();
        elapsed = finish - start;
        cout<<"ThirtyTwo Thread Algorithm Time:"<< elapsed.count() <<endl;
        cout<<resm.getElem(101,811)<<endl;
    }
}

void test02() { //生成1000*1000的两个矩阵相乘
    std::random_device rd;  // 使用随机设备作为种子
    std::mt19937 gen(rd()); // 使用 Mersenne Twister 引擎
    std::uniform_int_distribution<> dis(1, 10); // 生成范围在1到100之间的均匀分布的整数
    
    int r1=1000,c1=1000,r2=1000,c2=1000;
    int arrayA[r1*c1],arrayB[r2*c2];
    for(int i=0;i<1000;i++) {
        for(int j=0;j<1000;j++) {
            int pos = i*1000 + j;
            // 生成随机数
            arrayA[pos] = dis(gen);
            arrayB[pos] = dis(gen);
        }
    }
    
    Matrix<int> m1(r1,c1,arrayA);
    Matrix<int> m2(r2,c2,arrayB);
    
    cout<<"m1:"<<m1.getRows()<<" "<<m1.getCols()<<endl;
    cout<<"m2:"<<m2.getRows()<<" "<<m2.getCols()<<endl;

    // 基本的串行算法
    clock_t startTime,endTime;double basicTime;
    startTime = clock();
    Matrix<int> m3 = m1*m2;
    endTime = clock();
    basicTime = (double)(endTime-startTime)/CLOCKS_PER_SEC;
    cout<<"Serial Algorithm Time:"<<basicTime<<endl;
    // 使用openmp加速
    clock_t fStartTime,fEndTime;double fTime;
    fStartTime = clock();
    Matrix<int> m4 = m1.parallelMulti(m2,4);
    fEndTime = clock();
    fTime = (double)(fEndTime-fStartTime)/CLOCKS_PER_SEC;
    cout<<"Parallel Algorithm Time:"<<fTime<<endl;
}

void Serial() {
    std::random_device rd;  // 使用随机设备作为种子
    std::mt19937 gen(rd()); // 使用 Mersenne Twister 引擎
    std::uniform_int_distribution<> dis(1, 10); // 生成范围在1到100之间的均匀分布的整数

    for(int i=1000;i<=3000;i+=1000) {
        cout<<"=========="<<i<<"*"<<i<<"Matrix=========="<<endl;
        int r1=i,c1=i,r2=i,c2=i;
        int arrayA[r1*c1],arrayB[r2*c2];
        for(int j=0;j<r1;j++) {
            for(int k=0;k<c1;k++) {
                int pos = j*c1 + k;
                // 生成随机数
                arrayA[pos] = dis(gen);
                arrayB[pos] = dis(gen);
            }
        }
        Matrix<int> m1(r1,c1,arrayA);
        Matrix<int> m2(r2,c2,arrayB);
        Matrix<int> resm;
        auto start = std::chrono::high_resolution_clock::now();
        resm = m1*m2;
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        cout<<"Serial Algorithm Time:"<< elapsed.count() << endl;
    }
}