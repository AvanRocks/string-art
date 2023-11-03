#ifndef MATRIX_H
#define MATRIX_H

#include "cuda-fix.h"

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

#endif
