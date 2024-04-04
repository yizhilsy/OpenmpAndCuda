## <center>实验二  OpenMP并行编程实验报告</center>

<div style="float:right;font-size:20px">21121889 陆诗雨</div>

### 1.实验环境

- g++编译器`version 8.1.0 (x86_64-posix-sjlj-rev0, Built by MinGW-W64 project)`

- OpenMp并行编程库

- vscode编辑器

- Windows操作系统（Intel Core i7 11800H），Ubuntu操作系统（Intel(R) Core(TM) i7-10700），Ubuntu Vm虚拟机（Intel Core i7 11800H）

### 2.实验目的

- 掌握 OpenMP 并行编程基础；

- 掌握在 Linux 和 Windows平台上编译和运行 OpenMP 程序；

- 掌握如何使用 OpenMP 实现多线程编程并加速程序执行

### 3.实验步骤

根据学习资料进行openmp并行编程的实验，具体步骤及分析见第四节。

### 4.编程及实验结果分析（需要图表分析）

#### 4.1 HelloWorld程序，串行、并行但代码中不设置线程数（即默认线程），以及代码中设置8个线程，多执行几次，进行对比分析原因

##### 4.1.1 HelloWorld程序

```c++
#include <iostream>
#include <omp.h>
using namespace std;
int main()
{
    int nthreads,tid;
    omp_set_num_threads(8);// 通过是否使用这个函数来控制之后并行域中并行执行的线程数
    #pragma omp parallel private(nthreads,tid)
    {
        tid = omp_get_thread_num();
        printf("Hello World from OMP thread %d\n",tid);
        if(tid==0){
            nthreads = omp_get_num_threads();
            printf("Number of threads is %d\n",nthreads);
        }
    }
    return 0;
}
```

##### 4.1.2 OpenMp HelloWorld程序以默认线程执行

第一次执行默认线程数的并行HelloWorld程序

<img src="D:\typora\计算机体系结构\pics\Snipaste_2024-04-02_22-29-51.jpg" alt="Snipaste_2024-04-02_22-29-51" style="zoom:67%;" />

第二次执行默认线程数的并行HelloWorld程序

<img src="D:\typora\计算机体系结构\pics\Snipaste_2024-04-02_22-39-21.jpg" alt="Snipaste_2024-04-02_22-39-21" style="zoom:67%;" />

第三次执行默认线程数的并行HelloWorld程序

<img src="D:\typora\计算机体系结构\pics\Snipaste_2024-04-02_22-40-14.jpg" alt="Snipaste_2024-04-02_22-40-14" style="zoom:67%;" />

##### 4.1.3 OpenMp HelloWorld程序设置8个线程执行

第一次执行设置线程数为8的并行HelloWorld程序

<img src="D:\typora\计算机体系结构\pics\Snipaste_2024-04-02_22-32-03.jpg" alt="Snipaste_2024-04-02_22-32-03" style="zoom: 67%;" />

第二次执行设置线程数为8的并行HelloWorld程序

<img src="D:\typora\计算机体系结构\pics\Snipaste_2024-04-02_22-35-14.jpg" alt="Snipaste_2024-04-02_22-35-14" style="zoom: 80%;" />

第三次执行设置线程数为8的并行HelloWorld程序

<img src="D:\typora\计算机体系结构\pics\Snipaste_2024-04-02_22-37-20.jpg" alt="Snipaste_2024-04-02_22-37-20" style="zoom:80%;" />

##### 4.1.4 对比分析

在4.1.2中默认线程数执行HelloWorld程序时，由于本机是8核16线程的cpu，因此一开始的默认线程为16线程，所以在4.1.2中每次都有16个线程输出了结果。在调用了函数`omp_set_num_threads(8);`后，4.1.3中的并行域就设置为了8个线程同时执行，所以在4.1.3中每次都有8个线程输出了结果。

可以发现在4.1.2和4.1.3中，每次输出的结果都不一样，某次某个线程先输出，某次某个线程后输出。这对于并行编程是正常的现象，因为这取决于操作系统具体的调度。

#### 4.2 编程实现大规模向量的矩阵乘法并行计算

##### 4.2.1 生成1000$\times$1000、2000$\times$2000、3000$\times$3000的矩阵，矩阵元素可以是随机生成，也可以是等差数列。

本文使用了c++标准库中的`<random>`头文件中的函数来生成随机数，具体代码如下所示：

```c++
std::random_device rd;  // 使用随机设备作为种子
std::mt19937 gen(rd()); // 使用 Mersenne Twister 引擎
std::uniform_int_distribution<> dis(1, 10); // 生成范围在1到100之间的均匀分布的整数
```

此外，由于生成较大的矩阵及易发生栈溢出等异常，因此我们需要在编译语句中加入`-Wall -Wextra '-Wl,--stack=5368709120'`来扩展申请更大的栈空间，具体的g++编译语句如下：

```bash
g++ -o Openmp_MatrixMulti.exe Openmp_MatrixMulti.cpp -fopenmp -Wall -Wextra '-Wl,--stack=5368709120'
```

此外，本文为了精准计时并行域中的多线程加速的耗时，使用了c++标准库中的`chrono`头文件中的函数来计时，具体代码如下：

```c++
auto start = std::chrono::high_resolution_clock::now();
auto finish = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = finish - start;
cout<<"One Thread Algorithm Time:"<< elapsed.count() << endl;
```

对于传统串行矩阵乘法的实现本文不再赘述。本文对于矩阵乘法的多线程并行求解的代码如下：

```c++
template<typename ElemType>
Matrix<ElemType> Matrix<ElemType>::parallelMulti(Matrix<ElemType>& m1,const int threads) {
    const int resRows = this->rows;
    const int resCols = m1.cols;
    int* resArray = new int[resRows*resCols];
    // 定义每个线程计算的行数
    int OMP_THREAD_NUM = threads;
    int LINE_FOR_THREAD = (resRows+OMP_THREAD_NUM-1) /OMP_THREAD_NUM;
    // 数据量过小的情况
    if(LINE_FOR_THREAD==0) {
        OMP_THREAD_NUM = resRows;
        LINE_FOR_THREAD = 1;
    }
    // 并行计算加速
    #pragma omp parallel for num_threads(OMP_THREAD_NUM)
    for(int i=0;i<OMP_THREAD_NUM;i++) {
        // cout<<"NOW Thread ID:"<<omp_get_thread_num()<<endl;
        for(int line = i*LINE_FOR_THREAD;line<resRows && line<(i+1)*LINE_FOR_THREAD;line++) {
            for(int j=0;j<resCols;j++) {
                int value = 0;
                for(int k=0;k<(this->cols);k++) {
                    value += this->getElem(line,k) * m1.getElem(k,j);
                }
                int pos = line*resCols + j;
                resArray[pos] = value;
            }
        }
    }
    Matrix<ElemType> multiRes(resRows,resCols,resArray);
    delete[] resArray;
    return multiRes;
}
```

本文实现矩阵乘法的并行的主要思路是将第一个矩阵的行数按照设置的线程数平分（向上取整），由于矩阵乘法的结果矩阵两两元素互不影响，因此可以高效的实现并行。在此并行算法运算的结果都存在 `resArray` 数组中，之后根据手写的 `Matrix` 类构造函数即可生成结果矩阵。

##### 4.2.2 分别设置线程数为1、2、4、8、16、32画出不同线程数和运行时间的关系图，计算加速比（列表，可以对比Windows, Linux以及虚拟机下Linux系统中加速比的大小），分析和总结线程数和运行时间的关系及原因。

Windows操作系统下（单位：秒）

| 矩阵规模\线程数  | 串行程序 | 1       | 2       | 4       | 8        | 16       | 32       |
| ---------------- | -------- | ------- | ------- | ------- | -------- | -------- | -------- |
| 1000$\times$1000 | 2.79491  | 4.85656 | 2.37084 | 1.28329 | 0.893836 | 0.740115 | 0.701665 |
| 2000$\times$2000 | 35.3154  | 53.3997 | 27.7551 | 14.9153 | 8.79066  | 7.3727   | 6.69274  |
| 3000$\times$3000 | 163.969  | 236.047 | 106.43  | 57.617  | 34.5744  | 27.0443  | 26.3272  |

实验结果图片

![Snipaste_2024-04-03_23-34-44](D:\typora\计算机体系结构\pics\Snipaste_2024-04-03_23-34-44.jpg)

![Snipaste_2024-04-03_23-47-28](D:\typora\计算机体系结构\pics\Snipaste_2024-04-03_23-47-28.jpg)

不同问题规模不同线程耗时图

![Figure_1](D:\typora\计算机体系结构\pics\Figure_3.png)

不同问题规模不同线程加速比图

![Figure_5](D:\typora\计算机体系结构\pics\Figure_5.png)



Linux操作系统下

| 矩阵规模\线程数  | 串行程序 | 1       | 2       | 4       | 8        | 16       | 32       |
| ---------------- | -------- | ------- | ------- | ------- | -------- | -------- | -------- |
| 1000$\times$1000 | 3.40098  | 6.07884 | 3.04015 | 1.56173 | 0.812743 | 0.789141 | 0.833605 |
| 2000$\times$2000 | 42.6285  | 65.9972 | 32.7515 | 16.1433 | 8.7519   | 7.91376  | 8.43631  |
| 3000$\times$3000 | 195.225  | 279.569 | 136.186 | 68.691  | 38.6207  | 32.4448  | 33.0052  |

实验结果图片

<img src="D:\typora\计算机体系结构\pics\微信图片编辑_20240404185750.jpg" alt="微信图片编辑_20240404185750" style="zoom:50%;" />

不同问题规模不同线程耗时图

![Figure_7](D:\typora\计算机体系结构\pics\Figure_8.png)

不同问题规模不同线程加速比图

![Figure_9](D:\typora\计算机体系结构\pics\Figure_9.png)

设置16线程的Linux虚拟机下

| 矩阵规模\线程数  | 串行程序 | 1       | 2       | 4       | 8       | 16       | 32       |
| ---------------- | -------- | ------- | ------- | ------- | ------- | -------- | -------- |
| 1000$\times$1000 | 3.94829  | 5.88786 | 3.44634 | 1.77033 | 1.00514 | 0.771021 | 0.789357 |
| 2000$\times$2000 | 53.4981  | 73.7769 | 38.2722 | 19.6735 | 10.3403 | 7.48985  | 7.60986  |
| 3000$\times$3000 | 256.285  | 290.331 | 147.266 | 73.7636 | 38.2323 | 27.3949  | 28.0839  |

实验结果图片

![Snipaste_2024-04-04_00-21-25](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_00-21-25.jpg)

![Snipaste_2024-04-04_00-20-40](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_00-20-40.jpg)

不同问题规模不同线程耗时图

![Figure_2](D:\typora\计算机体系结构\pics\Figure_4.png)

不同问题规模不同线程加速比图

![Figure_6](D:\typora\计算机体系结构\pics\Figure_6.png)

在以上的三个OS系统中，我们发现了一些共性的规律。

- 当我们并行编程某块指定的域但只设置一个线程时，其运行时间长于普通串行程序。这是因为线程的创建和通信也是有代价的。
- 随着线程数的增加，并行程序执行的时间会随之减少（即有效加速了串行程序的运行），因为多线程成功开始并行处理程序了。但我们发现存在一个极限，此极限与本机处理器的线程数密切相关。例如在本实验中，由于处理器是16线程的，因此并行程序加速的极限就是在16-32线程那个点。甚至32线程有的时候慢于16线程，这是因为线程的创建是有代价的。
- 随着线程数的增加，相比于串行程序的加速比也在增加，其也有极限，与上面的第二点类似，超过那个极限后，加速比甚至可能会有轻微的下降。

#### 4.3 Openmp实例估算Pi值，调试PPT中串行算法及四种并行程序（需要做适当修改），检查四种并行程序是否有效提高加速比。想想其他性能优化方法：SSE2，动态调度，调度的策略

##### 4.3.1 计算Pi值的数学公式

$$
Pi = \int_{0}^{1}\frac{4}{1+x^2}dx = \frac{1}{N}\sum_{i=1}^{N}f(\frac{i}{N}-\frac{1}{2N}) = \frac{1}{N}\sum_{i=1}^{N}f(\frac{i-0.5}{N})
$$

##### 4.3.2 对比分析ppt中的串行算法以及四种并行程序能否有效提高加速比

本文将总步数设置为`1000000000（1e9）`,步长为步数的倒数。

###### 串行算法

```c++
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
```

实验结果

![Snipaste_2024-04-03_15-09-46](D:\typora\计算机体系结构\pics\Snipaste_2024-04-03_15-09-46.jpg)

串行程序即使用单核逐个遍历求和经过测试，串行程序在 Windows 下所需的时间为2.85398s。

###### 使用并行域并行化的程序

ppt原程序

```c++
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
    omp_set_num_threads(NUM_THREADS); //
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        double x;
        int id;
        id = omp_get_thread_num();
        for (i = id, sum[id] = 0.0; i < num_steps; i = i + NUM_THREADS)
        { //
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"The Pai Integer:"<<pi<<",time:"<<elapsed.count()<<endl;
    return 0;
}
```

实验结果

![Snipaste_2024-04-03_14-45-36](D:\typora\计算机体系结构\pics\Snipaste_2024-04-03_14-45-36.jpg)

这个程序显然是错误的。首先，其对于要并行处理的循环变量 `i` 没有作为每个线程私有的声明，其后果是每个线程都有对其修改的权利，造成了紊乱，原先是按照线程数的步长来递增，因此我们可以看到Pai值的计算是不正确的。从实验结果图中我们也可以看见其pai值计算错误，且耗时反而几乎是单线程的8倍（因为本文开了8个线程）。

==修改后的程序==（完整的测试程序详见附录）

顺着并行域并行化的思路，我们不妨将总步数按照开辟的线程数 `OMP_THREAD_NUM` 平分为 `OMP_THREAD_NUM` 块，每块需要执行 $\lceil numSteps/OMP\_THREAD\_NUM \rceil$ 个步数，把这个值记为 `LINE_FOR_THREADS` 。第i块负责计算$[i*LINE\_FOR\_THREADS,(i+1)*LINE\_FOR\_THREADS )$这个区间内的步数。最后将每个线程运算得到的结果相加即是结果。

```c++
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
    cout<<"The Pai Integer:"<<sum<<",time:"<<elapsed.count()<<endl;
}
```

实验结果

![Snipaste_2024-04-03_15-45-50](D:\typora\计算机体系结构\pics\Snipaste_2024-04-03_15-45-50.jpg)

可以发现，使用并行域并行化的方法后，在设置8个线程的情况下，运算效率得到了极大的提升，加速比为$\frac{2.86567}{0.895849} \approx 3.199$

###### 使用共享任务结构并行化的程序

ppt原程序

```c++
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
```

实验结果

![Snipaste_2024-04-03_19-13-31](D:\typora\计算机体系结构\pics\Snipaste_2024-04-03_19-13-31.jpg)

这个程序也是有问题的：一、我们在 `#pragma omp parallel` 设置的并行域中声明了for循环，二、i是共享变量，不同的线程在执行for循环的时候对i变量的访问可能导致并发和竞争条件。

虽然编译器会解析 `#pragma omp parallel` 并行域中的 `#pragma omp for` 声明并将 `for (i = 0; i < num_steps; i++)` 平均分给每个线程，但是由于循环变量 `i` 不是私有的，各个线程对其访问需要避免冲突，这造成了额外的时间开销，可以发现并行后所需的时间远远长于普通串行程序所需的时间。

==修改后的程序==（完整的测试程序详见附录）

事实上，要达到共享任务结构的目的，可以优化 `#pragma omp for` 这个声明，考虑直接采取归约 `reduction` 声明等手段即可达到加速的目的，其具体分析见本节**使用并行归约得出的并行程序**。

此外，我们还可以考虑使用任务队列法的方式。共享任务结构是指多个线程共享同一个任务队列，每个线程从队列中获取任务并执行。这种方式可以提高线程的利用率，因为当一个线程完成了它的任务时，它可以从共享队列中获取更多的任务来执行，而不必等待其他线程。

`#pragma omp task` 用于创建一个可以异步执行的任务。在并行程序中，任务是一种执行单元，可以在不同的线程中执行。使用任务可以实现细粒度的并行，将任务分配给可用的线程来执行，以充分利用多核处理器的并行性。具体来说，`#pragma omp task` 指令用于创建一个任务，该任务的执行被推迟到某个线程可用时执行。一旦任务被创建，它会进入任务队列，等待可用的线程来执行。

具体代码如下所示：

```c++
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
```

实验结果

![Snipaste_2024-04-03_21-41-35](D:\typora\计算机体系结构\pics\Snipaste_2024-04-03_21-41-35.jpg)

可以发现，使用任务队列后效率得到了很大的提升，加速比为$\frac{2.98055}{0.886925} \approx 3.361$

###### 使用`private`子句和`critical`部分并行化的程序

ppt源代码

```c++
#include <omp.h>
static long num_steps = 1000000000;
double step;
#define NUM_THREADS 2 
int main () 
{	  
	int i; 	  
	double x, sum, pi=0.0; 
	step = 1.0/(double) num_steps; 
	omp_set_num_threads(NUM_THREADS);
    auto start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel private (x, sum) 
	{	
		id = omp_get_thread_num(); 
	  	for (i=id,sum=0.0;i< num_steps;i=i+NUM_THREADS){ 
		  	x = (i+0.5)*step; 
		  	sum += 4.0/(1.0+x*x); 
	  	} 
		#pragma omp critical
	  		pi += sum 
	}
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"The Pai Integer:"<<pi<<",time:"<<elapsed.count()<<endl;
    return 0;
}
```

实验结果

![Snipaste_2024-04-04_00-52-29](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_00-52-29.jpg)

ppt源代码这里也是错误的，因为其对 `i` 这个循环变量每个线程不是私有的，这造成了每个线程循环过程的紊乱以及数据上的冲突，如下图所示，其不仅耗时，Pai值也没算对。

`#pragma omp critical` 是OpenMP中用来创建临界区（critical section）的指令。临界区是一段代码，在同一时间只允许一个线程进入执行，这样可以避免多线程访问共享资源时发生竞争条件（race condition）。`#pragma omp critical` 为每个线程创建一份指定变量的副本，同样避免多线程访问共享资源时发生竞争条件（race condition）。

==修改后的程序==（完整的测试程序详见附录）

```c++
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
```

实验结果

![Snipaste_2024-04-04_00-54-57](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_00-54-57.jpg)

可以发现，使用了critical和对 `i` 值private后效率得到了很大的提升，加速比为$\frac{2.94864}{0.897264} \approx 3.286$

###### 使用并行归约得出的并行程序

ppt源代码

```c++
#include <iostream>
#include <chrono>
#include <omp.h> 
using namespace std;
static long num_steps = 1000000000;
double step; 
#define NUM_THREADS 8
int repo[16];
int main () 
{	  
    int i; 	  
    double x, pi, sum = 0.0; 
    step = 1.0/(double) num_steps; 
    omp_set_num_threads(NUM_THREADS);
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum) private(x,i) 
    for (i=0;i<num_steps; i++){ 
        x = (i+0.5)*step; 
        sum = sum + 4.0/(1.0+x*x); 
        repo[omp_get_thread_num()] = i;
    } 
    pi = step * sum; 
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout<<"The Pai Integer:"<<pi<<",time:"<<elapsed.count()<<endl;
    for(int i=0;i<16;i++) {
        cout<<repo[i]<<endl;
    }

    return 0;
} 
```

实验结果

![Snipaste_2024-04-04_11-39-08](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_11-39-08.jpg)

当我们把线程数设置为8时，可以发现其计算出了正确的结果，而且并行也起到了加速的效果。然而，其实这种写法是不规范的，因为每个线程的 `i` 没有作私有变量的处理，从逻辑上讲会造成紊乱。但实际上，OpenMP 在处理循环迭代时会根据上下文自动推断哪些变量应该是私有的，哪些变量应该是共享的。在 `parallel for` 指令中，循环索引变量 `i` 通常会被默认声明为私有变量，以确保每个线程都有自己的独立副本，从而避免线程之间的竞争和数据冲突。但我们最好还是声明其为私有变量。

在 OpenMP 中，`reduction` 子句用于对多个线程中的私有变量进行归约操作，最常见的归约操作是求和、求积、求最大值、求最小值等。当多个线程同时更新一个全局变量时，可能会导致竞争条件和数据不一致性问题，使用 `reduction` 子句可以避免这些问题。在上面的例子中，我们使用了 `reduction(+:sum)` 这个子句，它的作用是将 `sum` 变量为每个线程创建一个副本，并让每个线程在执行完毕后将私有的 `sum` 变量累加完成后赋值给外部的 `sum` 变量中，保证不会产生冲突和竞态。

根据上面的分析，我们只需修改一小部分源程序即可，如下所示：

==修改后的程序==（完整的测试程序详见附录）

```c++
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
    cout<<"The Pai Integer:"<<sum<<",time:"<<elapsed.count()<<endl;
}
```

实验结果

![Snipaste_2024-04-04_12-12-09](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_12-12-09.jpg)

可以发现，使用了 `Reduction` 并行归约的方法后效率得到了很大的提升，加速比为$\frac{2.89244}{0.887306} \approx 3.260$



###### 其他性能优化方法

动态调度是一种OpenMP并行编程中的调度策略，它的好处主要体现在以下2个方面：

- **负载均衡：** 动态调度能够在运行时动态地将迭代任务分配给不同的线程，以确保各个线程的负载尽可能均衡。这可以避免某些线程负载过重而导致效率低下的情况，提高整体程序的性能。

- **灵活性：** 动态调度允许系统在运行时根据任务的实际情况进行任务分配和调度，具有较高的灵活性。这使得程序能够更好地适应不同的计算环境和工作负载，提高了程序的通用性和可移植性。

实验代码

```c++
// 使用动态调度
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
```

实验图片

![Snipaste_2024-04-04_17-45-22](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_17-45-22.jpg)

我们通过指定 `schedule(dynamic, 1000000)` 设定动态调度，通过实验结果也可以看见我们也成功优化了串行算法所需的时间，加速比为$\frac{2.83145}{0.915717} \approx 3.092$

#### 4.4 其他实验：包括私有变量和共有变量对比，并行化的额外负担（同步的开销，负载不均衡带来的影响），线程负载问题，线程同步问题

##### 4.4.1 私有变量和共有变量的对比

实验代码

```c++
#include <iostream>
#include <windows.h>
#include <omp.h>
int main() {
    int shared_variable = 0;
    int private_variable = 0;
    #pragma omp parallel shared(shared_variable) private(private_variable) num_threads(4)
    {
        private_variable = 0;
        int thread_id = omp_get_thread_num();
        // 修改共享变量和私有变量
        shared_variable += thread_id;
        private_variable += thread_id;
        #pragma omp critical
        // 打印每个线程的变量值
        std::cout << "Thread " << thread_id << ": Shared Variable = " << shared_variable
                  << ", Private Variable = " << private_variable << std::endl;
    }
    Sleep(2000);
    // 打印最终的变量值
    std::cout << "Final Values: Shared Variable = " << shared_variable
              << ", Private Variable = " << private_variable << std::endl;
    return 0;
}

```

实验结果

![Snipaste_2024-04-04_16-34-18](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_16-34-18.jpg)

我们可以发现，不同的线程下对于共有变量的修改是有效的，而对于私有变量，在线程内的修改不会同步到外部的变量。厘清这一点对于控制并行域内的循环很重要。

##### 4.4.2 并行化带来的额外负担

实验代码

```c++
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
```

实验结果

![Snipaste_2024-04-04_17-03-05](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_17-03-05.jpg)

从实验结果我们可以看出，在数据量规模较小的问题中，使用多线程并行化相比普通的串行更慢，虽然只是一点时间，但也足以说明并行化创建线程以及线程通信之间需要消耗的代价。

##### 4.4.3 负载不均衡带来的影响

实验代码

```c++
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
```

实验图片

![Snipaste_2024-04-04_17-15-31](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_17-15-31.jpg)

由于编译器对 `#pragma omo parallel for` 声明对不同线程分配循环次数是默认平分的，而我们的负载不均衡并行代码在 `i` 小于 `5000000` 时执行的是小任务，大于 `5000000` 执行的是大任务，因此不均衡；负载均衡的代码则是按照奇数偶数来划分，因此均衡。可以发现，负载不均衡相比负载均衡所需要耗费的时间更长，但多线程总体相比普通的串行执行效率还是更高的。因此我们编写并行程序的时候要尽量负载均衡。

##### 4.4.4 同步的开销

实验代码

```c++
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
```

实验图片

![Snipaste_2024-04-04_17-27-41](D:\typora\计算机体系结构\pics\Snipaste_2024-04-04_17-27-41.jpg)

通过这张图片我们可以发现使用 `#pragma omp critical` 的巨大同步开销，由于对共有变量 `sum` 是互斥访问的，必然会造成巨大的协调上的开销，因此使用 `#pragma omp parallel for reduction` 可以为每个线程创建一个副本来避免这样的问题。

### 5.实验感想

在进行了一系列OpenMP实验后，我对并行编程有了更深入的了解。通过这些实验，我体会到了并行化对程序性能的显著提升，但也意识到了一些潜在的挑战和限制。

首先，通过实验我发现，合理地利用OpenMP的并行机制可以大大加快程序的执行速度。比如，在对计算密集型任务进行并行化时，利用多线程可以有效地减少程序的运行时间。通过使用OpenMP的指令和工具，我能够轻松地将串行程序转换为并行程序，并且在不同的平台上实现跨平台的并行化。

此外，我意识到了负载不均衡对并行程序性能的影响。在某些情况下，负载不均衡会导致一些线程的执行时间远远超过其他线程，从而降低了整个程序的并行效率。为了解决这个问题，我学会了一些调整负载均衡的技巧，比如任务调度、工作划分等。通过实验我还了解到了OpenMP中共享变量和私有变量的区别。共享变量可以被所有线程访问和修改，但可能会引发竞争和同步的问题；而私有变量对每个线程都是独立的，不会受其他线程的影响，但需要额外的内存开销。

通过这些OpenMP实验，我对并行编程有了更深入的理解，并且学会了如何利用并行化技术来提高程序性能。我相信在今后的学习和工作中，这些经验会对我有所帮助。

### 6.附录（代码）

#### 6.1 矩阵类模板 Matrix.h

```c++
#include <iostream>
#include <omp.h>
using namespace std;

template <typename ElemType>
class Matrix {
private:
    int rows,cols;
    ElemType* ptr;

public:
    Matrix();
    Matrix(int r,int c,ElemType* dataPtr);
    ~Matrix();
    Matrix<ElemType>& operator=(const Matrix<ElemType>& m1);
    bool IsCanMulti(const Matrix<ElemType>& m1);
    // 矩阵乘法
    Matrix<ElemType> operator*(const Matrix<ElemType>& m1);
    // 并行矩阵乘法
    Matrix<ElemType> parallelMulti(Matrix<ElemType>& m1);
    // 打印矩阵
    // 友元函数重载<<运算符
    template<class T>
    friend ostream& operator<<(ostream& out,const Matrix<T>& m);
    
    ElemType& getElem(int row,int col);
    int getRows() const;
    int getCols() const;
    ElemType* getPtr() const;
};

// 矩阵类模板
template<typename ElemType>
Matrix<ElemType>::Matrix() {
    rows = 0;
    cols = 0;
    ptr = nullptr;
}

template<typename ElemType>
Matrix<ElemType>::Matrix(int r,int c,ElemType* dataPtr) {
    rows = r;
    cols = c;
    ptr = new ElemType[r*c];
    for(int i=0;i<r;i++) {
        for(int j=0;j<c;j++) {
            int nowPos = i*c + j;
            ptr[nowPos] = dataPtr[nowPos];
        }
    }
}

template<typename ElemType>
Matrix<ElemType>::~Matrix() {
    delete[] ptr;
}

template<typename ElemType>
Matrix<ElemType>& Matrix<ElemType>::operator= (const Matrix<ElemType>& m1) {
    if(ptr != nullptr) {
        delete[] ptr;
    }
    this->rows = m1.rows;
    this->cols = m1.cols;
    ptr = new ElemType[rows*cols];
    for(int i=0;i<rows;i++) {
        for(int j=0;j<cols;j++) {
            int nowPos = i*cols + j;
            ptr[nowPos] = m1.ptr[nowPos];
        }
    }
    return *(this);
}

template<typename ElemType>
bool Matrix<ElemType>::IsCanMulti(const Matrix<ElemType>& m1) {
    return this->cols==m1.rows;
}

template<typename ElemType>
Matrix<ElemType> Matrix<ElemType>::operator*(const Matrix<ElemType>& m1) {
    int resRows = this->rows;
    int resCols = m1.cols;
    int* resArray = new int[resRows*resCols];

    // #pragma omp parallel for schedule(dynamic, 100) num_threads(4)
    for(int i=0;i<resRows;i++) {
        // cout << "Thread " << omp_get_thread_num() << " processing iteration " << i << endl;
        for(int j=0;j<resCols;j++) {
            int value = 0;
            for(int k=0;k<(this->cols);k++) {
                value += this->ptr[i*(this->cols) + k] * m1.ptr[k*m1.cols + j];
            }
            resArray[i*resCols+j] = value;
        }
    }
    Matrix<ElemType> multiRes(resRows,resCols,resArray);
    delete[] resArray;
    return multiRes;
}

template<typename ElemType>
Matrix<ElemType> Matrix<ElemType>::parallelMulti(Matrix<ElemType>& m1) {
    int resRows = this->rows;
    int resCols = m1.cols;
    int* resArray = new int[resRows*resCols];

    // 定义每个线程计算的行数
    int OMP_THREAD_NUM = 4;
    int LINE_FOR_THREAD = (resRows+OMP_THREAD_NUM-1) /OMP_THREAD_NUM;

    // 数据量过小的情况
    if(LINE_FOR_THREAD==0) {
        OMP_THREAD_NUM = resRows;
        LINE_FOR_THREAD = 1;
    }

    // 并行计算加速
    #pragma omp parallel for num_threads(OMP_THREAD_NUM)
    for(int i=0;i<OMP_THREAD_NUM;i++) {
        cout<<"NOW Thread ID:"<<omp_get_thread_num()<<endl;
        for(int line = i*LINE_FOR_THREAD;line<resRows && line<(i+1)*LINE_FOR_THREAD;line++) {
            for(int j=0;j<resCols;j++) {
                int value = 0;
                for(int k=0;k<(this->cols);k++) {
                    value += this->getElem(line,k) * m1.getElem(k,j);
                }
                int pos = line*resCols + j;
                resArray[pos] = value;
            }
        }
    }
    Matrix<ElemType> multiRes(resRows,resCols,resArray);
    delete[] resArray;
    return multiRes;
}


template<class T>
ostream& operator<<(ostream& out,const Matrix<T>& m) {
    for(int i =0;i<m.rows;i++) {
        for(int j=0;j<m.cols;j++) {
            out << m.ptr[i*m.cols + j] << " ";
        }
        out << endl;
    }
    return out;
}

template<typename ElemType>
ElemType& Matrix<ElemType>::getElem(int row,int col) {
    return ptr[row*(this->cols) + col];
}

template<typename ElemType>
int Matrix<ElemType>::getRows() const {
    return this->rows;
}

template<typename ElemType>
int Matrix<ElemType>::getCols() const {
    return this->cols;
}

template<typename ElemType>
ElemType* Matrix<ElemType>::getPtr() const {
    return this->ptr;
}
```

#### 6.2 不同规模矩阵相乘测试运行时间的cpp文件

```c++
#include <iostream>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include "Matrix.h"
#include <chrono>
using namespace std;
void experiment();
void Serial();
int main()
{

    experiment();
    Serial();
    return 0;
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
```

#### 6.3 修改后的并行化计算Pai值的cpp文件（完整）

```c++
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
    cout<<"Using Reduction Schedule, The Pai Integer:"<<sum<<",time:"<<elapsed.count()<<endl;
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
    ParallelB();
    ParallelC();
    ParallelD();
    return 0;
}
```

