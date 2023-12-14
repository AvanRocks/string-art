#include "matrix.h"

const int BLOCK_SIZE = 768;

template<typename T1, typename T2>
__global__
void multiplyMatrixVector(const Matrix<Color> A, const T1 *v, T2 *res) {
	__shared__ T2 tmp[BLOCK_SIZE];
	int row = blockIdx.x;
	int rowStride = gridDim.x;
	for (; row < A.height; row += rowStride) {
		// each block does one dot product
		if (threadIdx.x == 0) {
			res[row] = T2(0);
		}
		int colStride = blockDim.x;
		for (int colStart = 0; colStart < A.width; colStart += colStride) {
			int colOffset = threadIdx.x;
			int col = colStart + colOffset;

			// do the multiplication
			if (col < A.width) {
				tmp[colOffset] = A.getElement(row, col) * v[col];
			} else {
				tmp[colOffset] = T2(0);
			}
			__syncthreads();

			// sum across tmp fast
			for (int n = 2; n / 2 < BLOCK_SIZE; n *= 2) {
				if (colOffset % n == 0) {
					if (colOffset + n / 2 < BLOCK_SIZE) {
						tmp[colOffset] += tmp[colOffset + n / 2];
					}
				}
				__syncthreads();
			}

			if (threadIdx.x == 0) {
				res[row] += tmp[0];
			}

			__syncthreads();
		}
	}
}

template __global__ void multiplyMatrixVector(const Matrix<Color>, const double *, Color *);
//template __global__ void multiplyMatrixVector(const Matrix<Color>, const Color *, double *);

__global__
void addVectors(const double *a, const double *b, unsigned int len, double *res) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int i = idx; i < len; i += stride) {
		res[i] = a[i] + b[i];
	}
}

__global__
void subtractVectors(const Color *a, const Color *b, unsigned int len, Color *res) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int i = idx; i < len; i += stride) {
		res[i] = a[i] - b[i];
	}
}

template<typename T>
__global__
void scaleVector(const T *v, unsigned int len, double scalar, T *res) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int i = idx; i < len; i += stride) {
		res[i] = scalar * v[i];
	}
}

template __global__ void scaleVector<double>(const double *, unsigned int, double, double *);
