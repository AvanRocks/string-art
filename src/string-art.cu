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
#include <fstream>

#include "string-art.h"
#include "image.h"
#include "color.h"
#include "matrix.h"
#include "stopwatch.h"
#include "debug-utils.h"
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

struct Point {
	int x, y;
};

vector<Point> precomputePegCoords(const int numPegs, const Image &img) {
	vector<Point> pegCoords;
	for (int peg = 0; peg < numPegs; peg++) {
		double theta = 2.0 * numbers::pi * peg / numPegs;
		double x = cos(theta);
		double y = sin(theta);

		unsigned width = img.getWidth();
		unsigned height = img.getHeight();

		// coordinates in the image (row and column)
		int imgX = round(((width - 1) / 2.0) * (1 + x));
		int imgY = round(((height - 1) / 2.0) * (1 - y));

		pegCoords.push_back({imgX, imgY});
	}
	return pegCoords;
}

void drawLine(Image &img, Point p, Point q, const Color &stringColor) {
	bool leftToRight = abs(p.x - q.x) > abs(p.y - q.y);
	if (leftToRight) {
		Point start = (p.x < q.x) ? p : q;
		Point end = (p.x < q.x) ? q : p;

		// slope of line from start to end
		double m = (double)(end.y - start.y) / (end.x - start.x);

		// draw the line to the canvas
		for (int x = start.x; x <= end.x; x++) {
			int y = round(m * (x - start.x) + start.y);
			img.setPixelColor(x, y, stringColor);
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
			img.setPixelColor(x, y, stringColor);
		}
	}
}

void generateMatrixA(Matrix<Color> &A, const Image &canvas, const vector<Point> pegCoords, const StringArtParams &params) {
	const unsigned IMAGE_RES = (int)ceil((double)canvas.getWidth() / params.rectSize); // assume square image

	for (int startPeg = 0, col = 0; startPeg < params.numPegs; startPeg++) {
		cout << (int)round((double)col / A.width * 100) << "%" << endl;
		for (int endPeg = startPeg + 1; endPeg < params.numPegs; endPeg++, col++) {
			Image tmp {canvas};

			Point startPos = pegCoords[startPeg];
			Point endPos = pegCoords[endPeg];

			//drawLine(tmp, startPos, endPos, params);
			drawLine(tmp, startPos, endPos, params.stringColor);

			if (endPeg == startPeg + 50) {
				//tmp.display();
			}

			// take the average color in a rectangular region
			for (int y = 0; y < IMAGE_RES; y++) {
				for (int x = 0; x < IMAGE_RES; x++) {
					int top = y * params.rectSize;
					int left = x * params.rectSize;
					int right = min(left + params.rectSize - 1, canvas.getWidth() - 1);
					int bottom = min(top + params.rectSize - 1, canvas.getHeight() - 1);
					Color totalColor {0};
					for (int yi = top; yi <= bottom; yi++) {
						for (int xi = left; xi <= right; xi++) {
							Color c = tmp.getPixelColor(xi, yi);
							totalColor += c;
						}
					}
					int row = y * IMAGE_RES + x;
					const int area = (bottom - top + 1) * (right - left + 1);
					Color c = totalColor / (double)area;

					// invert the color (assumed grayscale)
					// this is needed for the matrix approach to make sense
					// also assumed that the string color is black and the background color is white
					c = Color{1} - c;
					A.setElement(row, col, c);
				}
			}

		}
	}
}

void saveToCache(const Matrix<Color> &A, const StringArtParams &params, const Image &target) {
	ofstream newCache(params.matrixCacheFilename, ios::binary);

	if (newCache.fail()) {
		throw runtime_error("Failed to write to matrix cache file.");
	}

	const int WIDTH = target.getWidth();
	const int HEIGHT = target.getHeight();

	startStopwatch("writing to cache");
	newCache.write(reinterpret_cast<const char*>(&params.numPegs), sizeof(params.numPegs));
	newCache.write(reinterpret_cast<const char*>(&WIDTH), sizeof(WIDTH));
	newCache.write(reinterpret_cast<const char*>(&HEIGHT), sizeof(HEIGHT));
	newCache.write(reinterpret_cast<const char*>(&params.rectSize), sizeof(params.rectSize));
	newCache.write(reinterpret_cast<const char*>(A.elements), A.width * A.height * sizeof(Color));
	endStopwatch();
}

void makeStringArt(const StringArtParams& params) {
	params.validate();

	Image target {params.inputImageFilename};

	if (target.getWidth() != target.getHeight()) {
		throw invalid_argument("non-square image");
	}

	Image canvas {params.backgroundColor, target.getWidth(), target.getHeight()};

	if (params.grayscaleInput) {
		target.convertToGrayscale();
		target.write("tmp/grayscale.png");
	}

	vector<Point> pegCoords = precomputePegCoords(params.numPegs, target);

	const unsigned NUM_PAIRS = (params.numPegs * (params.numPegs - 1)) / 2;
	const unsigned IMAGE_RES = (int)ceil((double)target.getWidth() / params.rectSize); // assume square image
	const unsigned NUM_PIXELS = IMAGE_RES * IMAGE_RES;

	cout << "num pairs: " << NUM_PAIRS << endl;
	cout << "num pixels: " << NUM_PIXELS << endl;

	// matrix A
	const ull N = (ull)NUM_PIXELS * NUM_PAIRS;
	const ull SIZE = N * sizeof(Color);
	cout << "N: " << N << endl;
	cout << "SIZE: " << SIZE / 1e6 << " MB" << endl;
	Color *Adata;
	cudaAssert( cudaMallocManaged((void**)&Adata, SIZE) );
	Matrix<Color> A {NUM_PAIRS, NUM_PIXELS, Adata};

	ifstream cache(params.matrixCacheFilename, ios::binary);

	const int WIDTH = target.getWidth();
	const int HEIGHT = target.getHeight();

	if (cache.fail()) {
		startStopwatch("generating matrix A");
		generateMatrixA(A, canvas, pegCoords, params);
		endStopwatch();

		saveToCache(A, params, target);
	} else {
		// read from cache

		int cacheNumPegs, cacheWidth, cacheHeight, cacheRectSize;
		cache.read(reinterpret_cast<char*>(&cacheNumPegs), sizeof(cacheNumPegs));
		cache.read(reinterpret_cast<char*>(&cacheWidth), sizeof(cacheWidth));
		cache.read(reinterpret_cast<char*>(&cacheHeight), sizeof(cacheHeight));
		cache.read(reinterpret_cast<char*>(&cacheRectSize), sizeof(cacheRectSize));

		if (cacheNumPegs == params.numPegs && 
				cacheWidth == WIDTH &&
				cacheHeight == HEIGHT &&
				cacheRectSize == params.rectSize)
		{
			cache.read(reinterpret_cast<char*>(Adata), SIZE);
		} else {
			cout << "cache does not match current parameters. creating new cache" << endl;
			startStopwatch("generating matrix A");
			generateMatrixA(A, canvas, pegCoords, params);
			endStopwatch();
			saveToCache(A, params, target);
		}
	}

	cout << "bad elements of A:" << endl;
	for (int i = 0; i < A.height; i++) {
		for (int j = 0; j < A.width; j++) {
			Color c = A.getElement(i, j);
			if (!(c.red == c.green && c.green == c.blue)) {
				cout << i << ": " << c << endl;
			}
		}
	}


	startStopwatch("making A transpose");
	Color *Atdata;
	cudaAssert( cudaMallocManaged((void**)&Atdata, SIZE) );
	Matrix<Color> At {A.height, A.width, Atdata};
	for (int row = 0; row < At.height; row++) {
		for (int col = 0; col < At.width; col++) {
			At.setElement(row, col, A.getElement(col, row));
		}
	}
	endStopwatch();

	// starting solution vector
	double *x;
	cudaAssert( cudaMallocManaged((void**)&x, A.width * sizeof(double)) );
	for (int i = 0; i < A.width; i++) {
		x[i] = 0.5;
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

	// debugging
	vector<Color> bImgData;
	for (int i = 0; i < A.height; i++)
		bImgData.emplace_back(Color{1} - b[i]);
	Image bImg {bImgData.data(), IMAGE_RES, IMAGE_RES};
	bImg.write("tmp/target.png");

	// intermediate vector for gradient calculation
	Color *y;
	cudaAssert( cudaMallocManaged((void**)&y, A.height * sizeof(Color)) );

	// gradient vector
	double *grad;
	cudaAssert( cudaMallocManaged((void**)&grad, A.width * sizeof(double)) );

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
	//cout << "max threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << endl;
	//cout << "max threads per block: " << deviceProp.maxThreadsPerBlock << endl;
	//cout << "mutiprocessor count: " << deviceProp.multiProcessorCount << endl;

	const double initialLearningRate = 1.0/100;
	double learningRate = initialLearningRate;
	const int maxNumBlocks = deviceProp.maxThreadsPerMultiProcessor / BLOCK_SIZE * deviceProp.multiProcessorCount;
	const int numBlocksPart1 = min(maxNumBlocks, A.height);
	const int numBlocksPart2 = min(maxNumBlocks, max(1, A.height / BLOCK_SIZE));
	const int numBlocksPart3 = min(maxNumBlocks, At.height);
	const int numBlocksPart4 = min(maxNumBlocks, max(1, At.height / BLOCK_SIZE));
	cout << "numBlocksPart1: " << numBlocksPart1 << endl;
	cout << "numBlocksPart2: " << numBlocksPart2 << endl;
	cout << "numBlocksPart3: " << numBlocksPart3 << endl;
	cout << "numBlocksPart4: " << numBlocksPart4 << endl;
	cout << "maxNumBlocks: " << maxNumBlocks << endl;

	if (false)
	{
		multiplyMatrixVector<<<numBlocksPart1, BLOCK_SIZE>>>(A, x, y);
		cudaAssert( cudaPeekAtLastError() );
		cudaAssert( cudaDeviceSynchronize() );

		subtractVectors<<<numBlocksPart2, BLOCK_SIZE>>>(y, b, A.height, y);
		cudaAssert( cudaPeekAtLastError() );
		cudaAssert( cudaDeviceSynchronize() );

		vector<Color> yGood;
		for (int i = 0; i < A.height; i++) {
			Color c{0};
			for (int j = 0; j < A.width; j++) {
				c += A.getElement(i, j) * x[j];
			}
			yGood.emplace_back(c - b[i]);
		}

		if (yGood.size() != A.height) {
			cout << "different size" << endl;
			return;
		}

		bool same = 1;
		for (int i = 0; i < A.height; i++) {
			Color delta = yGood[i] - y[i];
			delta.red = abs(delta.red);
			delta.green = abs(delta.green);
			delta.blue = abs(delta.blue);
			if (delta.red > 1e-3 || delta.green > 1e-3 || delta.blue > 1e-3) {
				same = 0;
				cout << "they differ" << endl;
				cout << "yGood[" << i << "]: " << yGood[i] << endl;
				cout << "y[" << i << "]: " << y[i] << endl;
			}
		}
		if (same) cout << "all good" << endl;

		return;
	}

	vector<double> prevX(A.width);
	double cost = numeric_limits<double>::infinity();
	double prevCost = numeric_limits<double>::infinity();
	for (int iter = 0; iter < params.numIters; iter++) {
		multiplyMatrixVector<<<numBlocksPart1, BLOCK_SIZE>>>(A, x, y);
		cudaAssert( cudaPeekAtLastError() );
		cudaAssert( cudaDeviceSynchronize() );

		/*
		subtractVectors<<<numBlocksPart2, BLOCK_SIZE>>>(y, b, A.height, y);
		cudaAssert( cudaPeekAtLastError() );
		cudaAssert( cudaDeviceSynchronize() );
		*/

		for (int i = 0; i < A.height; i++) {
			y[i] -= b[i];
		}

		cost = 0;
		for (int i = 0; i < A.height; i++) {
			cost += y[i] * y[i];
		}
		cost /= A.height;
		if (iter % 10 == 0)
			cout << "cost: " << cost << endl;

		if (cost > prevCost) {
			cout << "reverting last step" << endl;
			// revert last step
			for (int i = 0; i < A.width; i++) {
				x[i] = prevX[i];
			}
			prevCost = numeric_limits<double>::infinity();
			// decrease learning rate
			learningRate *= 0.7;
			continue;
		}

		multiplyMatrixVector<<<numBlocksPart3, BLOCK_SIZE>>>(At, y, grad);
		cudaAssert( cudaPeekAtLastError() );
		cudaAssert( cudaDeviceSynchronize() );

		/*
		scaleVector<<<numBlocksPart4, BLOCK_SIZE>>>(grad, At.height, 2.0, grad);
		cudaAssert( cudaPeekAtLastError() );
		cudaAssert( cudaDeviceSynchronize() );
		*/

		for (int i = 0; i < At.height; i++) {
			grad[i] *= 2.0;
		}

		if (iter == 0) {
			cout << "initial:" << endl;
			// debugging
			vector<Color> yImgData;
			for (int i = 0; i < A.height; i++) {
				Color c = Color{1} - (y[i] + b[i]);
				c.red = clamp(c.red, 0., 1.);
				c.green = clamp(c.green, 0., 1.);
				c.blue = clamp(c.blue, 0., 1.);
				if (!(c.red == c.green && c.green == c.blue)) {
					cout << i << ": " << c << endl;
				}
				yImgData.emplace_back(c);
			}
			Image yImg {yImgData.data(), IMAGE_RES, IMAGE_RES};
			yImg.write("tmp/initial.png");

			Image initial{canvas};
			for (int startPeg = 0, idx = 0; startPeg < params.numPegs; startPeg++) {
				for (int endPeg = startPeg + 1; endPeg < params.numPegs; endPeg++, idx++) {
					Point startPos = pegCoords[startPeg];
					Point endPos = pegCoords[endPeg];
					//drawLine(initial, startPos, endPos, Color::interp(params.backgroundColor, params.stringColor, x[idx]));
					drawLine(initial, startPos, endPos, params.stringColor);
				}
			}

			// downsample ideal
			Image idealInitial {WHITE, IMAGE_RES, IMAGE_RES};
			for (int row = 0; row < IMAGE_RES; row++) {
				for (int col = 0; col < IMAGE_RES; col++) {
					int top = row * params.rectSize;
					int left = col * params.rectSize;
					int right = min(left + params.rectSize - 1, WIDTH - 1);
					int bottom = min(top + params.rectSize - 1, HEIGHT - 1);
					Color totalColor {0};
					for (int yi = top; yi <= bottom; yi++) {
						for (int xi = left; xi <= right; xi++) {
							Color c = initial.getPixelColor(xi, yi);
							totalColor += c;
						}
					}
					const int area = (bottom - top + 1) * (right - left + 1);
					Color c = totalColor / area;


					// invert colors
					//c = Color{1} - c;
					//b[row * IMAGE_RES + col] = c;
					idealInitial.setPixelColor(col, row, c);
				}
			}
			idealInitial.write("tmp/ideal-initial.png");
		
		}

		if (iter % 10 == 0) {
			double mag = 0;
			for (int i = 0; i < A.width; i++) {
				mag += grad[i] * grad[i];
			}
			mag = sqrt(mag) / A.width;
			cout << "mag: " << mag << endl;
		}

		learningRate *= 1.1;

		for (int i = 0; i < A.width; i++) {
			x[i] -= learningRate * grad[i];
			if (x[i] < 0) x[i] = 0;
			if (x[i] > 1) x[i] = 1;
		}

		prevCost = cost;
		for (int i = 0; i < A.width; i++) {
			prevX[i] = x[i];
		}
		//cout << "done " << iter + 1 << endl;
	}

	// debugging
	cout << "product:" << endl;
	vector<Color> yImgData;
	for (int i = 0; i < A.height; i++) {
			//yImgData.emplace_back(Color{1} - (y[i] + b[i]));
			Color c = Color{1} - (y[i] + b[i]);
			c.red = clamp(c.red, 0., 1.);
			c.green = clamp(c.green, 0., 1.);
			c.blue = clamp(c.blue, 0., 1.);
			if (!(c.red == c.green && c.green == c.blue)) {
				cout << i << ": " << c << endl;
			}
			yImgData.emplace_back(c);
	}
	Image yImg {yImgData.data(), IMAGE_RES, IMAGE_RES};
	yImg.write("tmp/product.png");

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


	//cout << "x" << endl;
	//for (int i = 0; i < A.width; i++) cout << i << ": " << x[i] << endl;

	/*
	for (int startPeg = 0, idx = 0; startPeg < params.numPegs; startPeg++) {
		for (int endPeg = startPeg + 1; endPeg < params.numPegs; endPeg++, idx++) {
			if (x[idx] > 0.3) {
				Point startPos = pegCoords[startPeg];
				Point endPos = pegCoords[endPeg];
				drawLine(canvas, startPos, endPos, Color::interp(params.backgroundColor, params.stringColor, x[idx]));
			}
		}
	}
	canvas.write(params.outputImageFilename);
	*/
		
	/*
	for (int startPeg = 0, idx = 0; startPeg < params.numPegs; startPeg++) {
		for (int endPeg = startPeg + 1; endPeg < params.numPegs; endPeg++, idx++) {
			Point startPos = pegCoords[startPeg];
			Point endPos = pegCoords[endPeg];
			drawLine(canvas, startPos, endPos, Color::interp(params.backgroundColor, params.stringColor, x[idx]));
		}
	}
	canvas.write(params.outputImageFilename);
	*/
		
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
			Point startPos = pegCoords[startPeg];
			Point endPos = pegCoords[endPeg];
			drawLine(tmp, startPos, endPos, params.stringColor);
		}
		cout << " done" << endl;
		
		cout << "writing ...";
		tmp.write(params.outputImageFilename);
		cout << " done" << endl;
	}

	cudaFree(x);
}

