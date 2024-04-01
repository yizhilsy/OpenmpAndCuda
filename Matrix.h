#include <iostream>
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
    // 打印矩阵
    // 重载<<

    template<class T>
    friend ostream& operator<<(ostream& out,const Matrix<T>& m);
    
    ElemType getElem(int row,int col);
    int getRows();
    int getCols();
    ElemType* getPtr();
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
ElemType Matrix<ElemType>::getElem(int row,int col) {
    return ptr[row*(this->cols) + col];
}

template<typename ElemType>
int Matrix<ElemType>::getRows() {
    return this->rows;
}

template<typename ElemType>
int Matrix<ElemType>::getCols() {
    return this->cols;
}

template<typename ElemType>
ElemType* Matrix<ElemType>::getPtr() {
    return this->ptr;
}