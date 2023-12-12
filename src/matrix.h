#ifndef MATRIX_H
#define MATRIX_H

#include "cuda-fix.h"
#include "color.h"

// Matrix on a GPU
template <typename T>
class Matrix {
public:
	Matrix(unsigned width, unsigned height, T *elements)
		: width{width}
		, height{height}
		, elements{elements}
		{}

	__host__ __device__
	T getElement(unsigned row, unsigned col) const {
		return elements[row * width + col];
	}

	void setElement(unsigned row, unsigned col, const T &val) {
		elements[row * width + col] = val;
	}

	const unsigned width, height;
	T * const elements;
private:
};

extern const int BLOCK_SIZE;

template<typename T1, typename T2>
__global__
extern void multiplyMatrixVector(const Matrix<Color> A, const T1 *v, T2 *res);

__global__
void addVectors(const Color *a, const Color *b, unsigned int len, Color *res);

__global__
void subtractVectors(const Color *a, const Color *b, unsigned int len, Color *res);

template<typename T>
__global__
extern void scaleVector(const T *v, unsigned int len, double scalar, T *res);

#endif
