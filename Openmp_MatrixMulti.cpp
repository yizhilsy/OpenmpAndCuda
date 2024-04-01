#include <iostream>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include "Matrix.h"
using namespace std;

void test01();
void test02();


int main()
{
    test01();
    cout<<"**************************"<<endl;
    test02();
    
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
    cout<<m3<<endl;
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
    cout<<"基本的串行算法时间："<<basicTime<<endl;

    // 使用openmp加速
    int ans[1000][1000];
    startTime = clock();

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < r1; ++i) {
            for (int j = 0; j < c2; ++j) {
                int value = 0;
                for (int k = 0; k < c1; ++k) {
                    value += m1.getElem(i,k)*m2.getElem(k,j);
                }
                ans[i][j] = value;
            }
        }
    }
    
    endTime = clock();
    basicTime = (double)(endTime-startTime)/CLOCKS_PER_SEC;
    cout<<"并行算法时间："<<basicTime<<endl;
}