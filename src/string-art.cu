#include <iostream>
#include <cmath>
#include <set>
#include <queue>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <cstdio>
#include <iomanip>
#include <vector>

#include "string-art.h"
#include "image.h"
#include "bit.h"
#include "color.h"
#include "matrix.h"
using namespace std;

typedef unsigned long long ull;

#define cudaAssert(ans) { cudaAssertFunc((ans), __FILE__, __LINE__); }
inline void cudaAssertFunc(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Point {
	int x, y;
};

struct Coords {
	double x, y;
};

void StringArtParams::validate() const {
	if (this->inputImageFilename.size() == 0) {
		throw logic_error("Input image filename is empty.");
	}
}

double flatThickness(double distance, unsigned sideLength) {
	const double THICKNESS = 1;
	const double END = 0.5 / 768 * sideLength;
	if (0 <= distance && distance <= END) {
		return THICKNESS;
	} else {
		return 0;
	}
}

double trapezoidThickness(double distance, unsigned sideLength) {
	const double MAX_THICKNESS = 1;
	const double FADE_START = max(0.2 / 768 * sideLength, 0.2);
	const double FADE_END = max(0.5 / 768 * sideLength, 0.4);
	if (0 <= distance && distance <= FADE_START) {
		return MAX_THICKNESS;
	} else if (distance <= FADE_END) {
		return  (MAX_THICKNESS / (FADE_START - FADE_END)) * (distance - FADE_END);
	} else {
		return 0;
	}
}

double absDistanceCost(const Color& c1, const Color& c2) {
	double dRed = static_cast<double>(c1.red) - c2.red;
	double dGreen = static_cast<double>(c1.green) - c2.green;
	double dBlue = static_cast<double>(c1.blue) - c2.blue;
	return abs(dRed) + abs(dGreen) + abs(dBlue);
}

double euclideanDistanceCost(const Color& c1, const Color& c2) {
	double dRed = static_cast<double>(c1.red) - c2.red;
	double dGreen = static_cast<double>(c1.green) - c2.green;
	double dBlue = static_cast<double>(c1.blue) - c2.blue;
	return sqrt( pow(dRed, 2) + pow(dGreen, 2) + pow(dBlue, 2) );
}

// plane coords refers to usual coordinates in the cartesian plane (normalized to [-1, 1])
// image coords means the row and column in the image
// x and y are [-1, 1] 
Point planeToImageCoords(double x, double y, const Image &img) {
	return { (int)round(((img.getWidth() - 1) / 2.0) * (1 + x)),
					 (int)round(((img.getHeight() - 1) / 2.0) * (1 - y)) };
}

Coords imageToPlaneCoords(int x, int y, const Image &img) {
	return { ((double)x + 0.5) *  (2.0 / img.getWidth()) - 1,
					 ((double)y + 0.5) *  (2.0 / img.getHeight()) - 1 };

}

// get the coordinates of the nth peg (0-indexed)
Point getPegCoords(int n, int N, const Image &img) {
	double theta = 2.0 * numbers::pi * n / N;
	double x = cos(theta);
	double y = sin(theta);

	unsigned width = img.getWidth();
	unsigned height = img.getHeight();

	return planeToImageCoords(x, y, img);
}

bool useBit = false;

void drawLine(Image &img, Point p, Point q, const StringArtParams &params, array<Bit<double>, 3> *bitsPtr = nullptr) {
	bool leftToRight = abs(p.x - q.x) > abs(p.y - q.y);
	if (leftToRight) {
		Point start = (p.x < q.x) ? p : q;
		Point end = (p.x < q.x) ? q : p;

		// slope of line from start to end
		double m = (double)(end.y - start.y) / (end.x - start.x);

		// draw the line to the canvas
		for (int x = start.x; x <= end.x; x++) {
			int y = round(m * (x - start.x) + start.y);
			if (useBit) {
				Color orig = img.getPixelColor(x, y);
				array<Bit<double>, 3> &bits = *bitsPtr;
				bits[0].update(x, y, params.stringColor.red - orig.red);
				bits[1].update(x, y, params.stringColor.green - orig.green);
				bits[2].update(x, y, params.stringColor.blue - orig.blue);
			}
			//img.setPixelColor(x, y, params.stringColor);
		}

	} else {
		// top to bottom
		Point start = (p.y < q.y) ? p : q;
		Point end = (p.y < q.y) ? q : p;

		// reciprocal of slope of line from start to end
		double mInv = (double)(end.x - start.x) / (end.y - start.y);

		// draw the line to the canvas
		for (int y = start.y; y <= end.y; y++) {
			int x = round(mInv * (y - start.y) + start.x);
			if (useBit) {
				Color orig = img.getPixelColor(x, y);
				array<Bit<double>, 3> &bits = *bitsPtr;
				bits[0].update(x, y, params.stringColor.red - orig.red);
				bits[1].update(x, y, params.stringColor.green - orig.green);
				bits[2].update(x, y, params.stringColor.blue - orig.blue);
			}
			img.setPixelColor(x, y, params.stringColor);
		}
	}
}

const int threadsPerBlock = 768;

__global__
void computeGradientPart1(const Matrix<Color> A, const double *x, const Color *b, Color *y, const Color STRING_COLOR) {
	int row = blockIdx.x;
	int rowStride = gridDim.x;
	for (; row < A.height; row += rowStride) {
		if (threadIdx.x == 0) {
			y[row] = Color{0};
		}
		__syncthreads();

		// consider groups of blockDim.x threads
		int colStride = blockDim.x;
		for (int colStart = 0; colStart < A.width; colStart += colStride) {
			int colOffset = threadIdx.x;
			int col = colStart + colOffset;
			__shared__ Color tmp[threadsPerBlock];

			// do the multiplication
			if (col < A.width) {
				tmp[colOffset] = A.getElement(row, col) * x[col];
			}
			__syncthreads();

			// sum across tmp fast
			for (int n = 2; n <= colStride; n *= 2) {
				if (colOffset < colStride / n) {
					tmp[colOffset] += tmp[colOffset + n / 2];
				}
				__syncthreads();
			}

			if (threadIdx.x == 0) {
				y[row] += tmp[0];
			}
		}

		__syncthreads();
		if (threadIdx.x == 0) {
			y[row] -= b[row];
		}
		__syncthreads();
	}
}

__global__
void computeGradientPart2(const Matrix<Color> A, const Color *y, double *grad) {
	int col = blockIdx.x;
	int colStride = gridDim.x;
	for (; col < A.width; col += colStride) {
		if (threadIdx.x == 0) {
			grad[col] = 0;
		}
		__syncthreads();

		int rowStride = blockDim.x;
		for (int rowStart = 0; rowStart < A.height; rowStart += rowStride) {
			int rowOffset = threadIdx.x;
			int row = rowStart + rowOffset;
			__shared__ double tmp[threadsPerBlock];

			// do the multiplication
			if (row < A.height) {
				tmp[rowOffset] = y[row] * A.getElement(row, col);
			}
			__syncthreads();


			// sum across tmp fast
			for (int n = 2; n <= rowStride; n *= 2) {
				if (rowOffset < rowStride / n) {
					tmp[rowOffset] += tmp[rowOffset + n / 2];
				}
				__syncthreads();
			}

			__syncthreads();
			if (threadIdx.x == 0) {
				grad[col] += tmp[0];
			}
			__syncthreads();
		}

		__syncthreads();
		if (threadIdx.x == 0) {
			grad[col] *= 2;
		}
		__syncthreads();
	}
}

void makeStringArt(const StringArtParams& params) {
	params.validate();

	Image target {params.inputImageFilename};
	Image canvas {params.backgroundColor, target.getWidth(), target.getHeight()};

	if (params.grayscaleInput) {
		target.convertToGrayscale();
		target.write("tmp/grayscale.png");
	}

	const unsigned NUM_PEGS = params.numPegs;
	const unsigned NUM_PAIRS = (NUM_PEGS * (NUM_PEGS - 1)) / 2;
	const unsigned IMAGE_RES = (int)ceil((double)target.getWidth() / params.rectSize); // assume square image
	const unsigned NUM_PIXELS = IMAGE_RES * IMAGE_RES;

	cout << "num pairs: " << NUM_PAIRS << endl;
	cout << "num pixels: " << NUM_PIXELS << endl;

	const ull N = (ull)NUM_PIXELS * NUM_PAIRS;
	const ull SIZE = N * sizeof(Color);
	cout << "N: " << N << endl;
	cout << "SIZE: " << SIZE / 1e6 << " MB" << endl;

	// matrix A
	Color *Adata;
	cudaAssert( cudaMallocManaged((void**)&Adata, SIZE) );
	Matrix<Color> A {NUM_PAIRS, NUM_PIXELS, Adata};

	cout << "preprocessing ..." << endl;

	if (useBit) {
		// make the bits
		unsigned width = canvas.getWidth();
		unsigned height = canvas.getHeight();
		array<Bit<double>, 3> origBits {{{width, height}, {width, height}, {width, height}}};
		for (unsigned y = 0; y < height; y++) {
			for (unsigned x = 0; x < width; x++) {
				Color c = canvas.getPixelColor(x, y);
				origBits[0].update(x, y, c.red);
				origBits[1].update(x, y, c.green);
				origBits[2].update(x, y, c.blue);
			}
		}
	}

	const int WIDTH = target.getWidth();
	const int HEIGHT = target.getHeight();
	for (int startPeg = 0, col = 0; startPeg < params.numPegs; startPeg++) {
		cout << (int)round((double)col / A.width * 100) << "%" << endl;
		for (int endPeg = startPeg + 1; endPeg < params.numPegs; endPeg++, col++) {
			Image tmp {canvas};

			Point startPos = getPegCoords(startPeg, params.numPegs, tmp);
			Point endPos = getPegCoords(endPeg, params.numPegs, tmp);

			if (useBit) {
				//array<Bit<double>, 3> bits {origBits};
				//drawLine(tmp, startPos, endPos, params, &bits);
			} else {
				drawLine(tmp, startPos, endPos, params);
			}

			if (endPeg == startPeg + 50) {
				//tmp.display();
			}

			for (int y = 0; y < IMAGE_RES; y++) {
				for (int x = 0; x < IMAGE_RES; x++) {
					int top = y * params.rectSize;
					int left = x * params.rectSize;
					int right = min(left + params.rectSize - 1, WIDTH - 1);
					int bottom = min(top + params.rectSize - 1, HEIGHT - 1);
					Color totalColor {0};
					for (int yi = top; yi <= bottom; yi++) {
						for (int xi = left; xi <= right; xi++) {
							Color c = tmp.getPixelColor(xi, yi);
							totalColor += c;
						}
					}
					int row = y * IMAGE_RES + x;
					const int area = (bottom - top + 1) * (right - left + 1);
					Color c = totalColor / area;

					// invert the color (assumed grayscale)
					// this is needed for the matrix approach to make sense
					// also assumed that the string color is black and the background color is white
					c = Color{1} - c;
					A.setElement(row, col, c);
				}
			}

		}
	}

	// debugging
	/*
	Image matrixA {A.elements, A.width, A.height};
	matrixA.write("tmp/test-matrix-A.png");
	*/

	cout << "done" << endl;

	// starting solution vector
	double *x;
	cudaAssert( cudaMallocManaged((void**)&x, A.width * sizeof(double)) );
	for (int i = 0; i < A.width; i++) {
		x[i] = 0;
	}

	// target image vector
	Color *b;
	cudaAssert( cudaMallocManaged((void**)&b, A.height * sizeof(Color)) );
	for (int row = 0; row < IMAGE_RES; row++) {
		for (int col = 0; col < IMAGE_RES; col++) {
			int top = row * params.rectSize;
			int left = col * params.rectSize;
			int right = min(left + params.rectSize - 1, WIDTH - 1);
			int bottom = min(top + params.rectSize - 1, HEIGHT - 1);
			Color totalColor {0};
			for (int yi = top; yi <= bottom; yi++) {
				for (int xi = left; xi <= right; xi++) {
					Color c = target.getPixelColor(xi, yi);
					totalColor += c;
				}
			}
			const int area = (bottom - top + 1) * (right - left + 1);
			Color c = totalColor / area;
			// invert colors
			c = Color{1} - c;
			b[row * IMAGE_RES + col] = c;
		}
	}

	// intermediate vector for gradient calculation
	Color *y;
	cudaAssert( cudaMallocManaged((void**)&y, A.height * sizeof(Color)) );

	// gradient vector
	double *grad;
	cudaAssert( cudaMallocManaged((void**)&grad, A.width * sizeof(double)) );

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
	cout << "max threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << endl;
	cout << "max threads per block: " << deviceProp.maxThreadsPerBlock << endl;
	cout << "mutiprocessor count: " << deviceProp.multiProcessorCount << endl;

	const double scalingFactor = 1.0/50;
	const int maxNumBlocks = deviceProp.maxThreadsPerMultiProcessor / threadsPerBlock * deviceProp.multiProcessorCount;
	const int numBlocksPart1 = min(maxNumBlocks, A.height);
	const int numBlocksPart2 = min(maxNumBlocks, A.width);
	cout << "numBlocksPart1: " << numBlocksPart1 << endl;
	cout << "numBlocksPart2: " << numBlocksPart2 << endl;
	cout << "maxNumBlocks: " << maxNumBlocks << endl;

	for (int iter = 0; iter < params.numIters; iter++) {
		computeGradientPart1<<<numBlocksPart1, threadsPerBlock>>>(A, x, b, y, params.stringColor);
		cudaAssert( cudaPeekAtLastError() );
		cudaAssert( cudaDeviceSynchronize() );

		//cout << "y" << endl;
		//for (int i = 0; i < A.height; i+=50) cout << i << ": " << y[i] << endl;

		computeGradientPart2<<<numBlocksPart2, threadsPerBlock>>>(A, y, grad);
		cudaAssert( cudaPeekAtLastError() );
		cudaAssert( cudaDeviceSynchronize() );

		cout << "gradient" << endl;
		//for (int i = 0; i < A.width; i+=200) cout << i << ": " << grad[i] << endl;
		int cntNeg = 0;
		double maxGrad = INT_MIN, minGrad = INT_MAX;
		double mag = 0;
		for (int i = 0; i < A.width; i++) {
			if (grad[i] < 0) cntNeg++;
			maxGrad = max(maxGrad, grad[i]);
			minGrad = min(minGrad, grad[i]);
			mag += grad[i] * grad[i];
		}
		mag = sqrt(mag);
		cout << "cntNeg: " << cntNeg << endl;
		cout << "maxGrad: " << maxGrad << endl;
		cout << "minGrad: " << minGrad << endl;
		cout << "mag: " << mag << endl;

		for (int i = 0; i < A.width; i++) {
			x[i] -= scalingFactor * grad[i];
			if (x[i] < 0) x[i] = 0;
			if (x[i] > 1) x[i] = 1;
		}

		cout << "done " << iter+1 << endl;
	}

	/*
	int cntz=0;
	for (int i = 0; i < A.width; i++) if (x[i] <= 0) cntz++;
	cout << "cntz: " << cntz << endl;
	cout << "x" << endl;
	for (int i = 0; i < A.width; i+=200) cout << i << ": " << x[i] << endl;
	cout << "y" << endl;
	for (int i = 0; i < A.height; i++) cout << i << ": " << y[i] << endl;
	cout << "gradient" << endl;
	for (int i = 0; i < A.width; i++) cout << i << ": " << grad[i] << endl;
	*/

	cudaFree(A.elements);
	cudaFree(Adata);
	cudaFree(b);
	cudaFree(y);
	cudaFree(grad);

	while (true) {
		cout << "enter num lines: ";
		int numLines = 0;
		cin >> numLines;
		if (!(0 <= numLines && numLines <= A.width)) {
			cout << "invalid num lines" << endl;
			continue;
		}

		cout << "drawing ...";
		Image tmp {canvas};
		vector<pair<double,pair<int,int>>> v;
		for (int startPeg = 0, idx = 0; startPeg < params.numPegs; startPeg++) {
			for (int endPeg = startPeg + 1; endPeg < params.numPegs; endPeg++, idx++) {
				v.push_back({x[idx], {startPeg, endPeg}});
			}
		}
		sort(v.rbegin(), v.rend());
		for (int i = 0; i < numLines && i < v.size(); i++) {
			int startPeg, endPeg;
			tie(startPeg, endPeg) = v[i].second;
			//cout << startPeg << " " << endPeg << endl;
			Point startPos = getPegCoords(startPeg, params.numPegs, tmp);
			Point endPos = getPegCoords(endPeg, params.numPegs, tmp);
			drawLine(tmp, startPos, endPos, params);
		}
		cout << " done" << endl;
		
		cout << "writing ...";
		tmp.write(params.outputImageFilename);
		cout << " done" << endl;
	}

	cudaFree(x);

	/*
	if (params.grayscaleInput) {
		img.convertToGrayscale();
	}
	img.write("tmp/grayscale.png");

	if (params.rgbOutput) {
		// do three passes, one for each color
		StringArtParams newParams {params};
		newParams.stringColor = Color{1, 0, 0};
		draw(img, output, newParams);
		newParams.stringColor = Color{0, 1, 0};
		draw(img, output, newParams);
		newParams.stringColor = Color{0, 0, 1};
		draw(img, output, newParams);
	} else {
		// single pass with specified color
		draw(img, output, params);
	}

	output.write(params.outputImageFilename);
	*/
}

