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
