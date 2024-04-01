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
    Matrix<ElemType> parallelMulti(const Matrix<ElemType>& m1);
    // 打印矩阵
    // 友元函数重载<<运算符
    template<class T>
    friend ostream& operator<<(ostream& out,const Matrix<T>& m);
    
    ElemType getElem(int row,int col) const;
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
    int ct=0;
    for(int i=0;i<resRows;i++) {
        for(int j=0;j<resCols;j++) {
            int value = 0;
            for(int k=0;k<(this->cols);k++) {
                value += this->ptr[i*(this->cols) + k] * m1.ptr[k*m1.cols + j];
            }
            resArray[ct++] = value;
        }
    }
    Matrix<ElemType> multiRes(resRows,resCols,resArray);
    delete[] resArray;
    return multiRes;
}

template<typename ElemType>
Matrix<ElemType> Matrix<ElemType>::parallelMulti(const Matrix<ElemType>& m1) {
    int resRows = this->rows;
    int resCols = m1.cols;
    int* resArray = new int[resRows*resCols];

    // 定义每个线程计算的行数
    int OMP_THREAD_NUM = 4;
    int LINE_FOR_THREAD = resRows/OMP_THREAD_NUM;

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
ElemType Matrix<ElemType>::getElem(int row,int col) const {
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