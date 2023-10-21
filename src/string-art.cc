#include <iostream>
#include <cmath>
#include <set>
#include <queue>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <array>

#include "string-art.h"
#include "image.h"
#include "bit.h"
using namespace std;

struct Point {
	int x, y;
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

// get the coordinates of the nth peg (0-indexed)
Point getPegCoords(int n, int N, const Image &img) {
	double theta = 2.0 * numbers::pi * n / N;
	double x = cos(theta);
	double y = sin(theta);

	unsigned width = img.getWidth();
	unsigned height = img.getHeight();

	int imgX = round(((width - 1) / 2.0) * (1 + x));
	int imgY = round(((height - 1) / 2.0) * (1 - y));

	return { imgX, imgY };
}

bool useBit = true;

// calculate the improvement that would be gained by drawing a line
double calcImprovement(
		const Image &reference,
		const array<Bit<double>, 3> &referenceBits,
		const Image &canvas,
		const array<Bit<double>, 3> &canvasBits,
		const Point &pos1,
		const Point &pos2,
		const StringArtParams &params
		) 
{

	double totalImprovement = 0;

	// number of rectangles visited
	int numVisited = 0;

	bool leftToRight = abs(pos1.x - pos2.x) > abs(pos1.y - pos2.y);

	if (leftToRight) {
		Point start = (pos1.x < pos2.x) ? pos1 : pos2;
		Point end = (pos1.x < pos2.x) ? pos2 : pos1;

		// slope of line from start to end
		double m = (double)(end.y - start.y) / (end.x - start.x);
		int width = reference.getWidth();
		int height = reference.getHeight();

		// visit each rectangle
		for (int x1 = start.x, x2 = 0; x1 < end.x; x1 = x2, numVisited++) {
			int y1 = round(m * (x1 - start.x) + start.y);
			x2 = min(x1 + params.rectSize, end.x);
			int y2 = round(m * (x2 - start.x) + start.y);
			int midX = (x1 + x2) / 2;
			int midY = (y1 + y2) / 2;
			int boxLeft = max(midX - params.rectSize / 2, 0);
			int boxRight = min(midX + params.rectSize / 2, width - 1);
			int boxTop = max(midY - params.rectSize / 2, 0);
			int boxBottom = min(midY + params.rectSize / 2, height - 1);

			// calculate average color in this rectangle in the reference and canvas images
			Color referenceTotal(0), canvasTotal(0);
			if (useBit) {
				referenceTotal.red += referenceBits[0].query(boxLeft, boxRight, boxTop, boxBottom);
				referenceTotal.green += referenceBits[1].query(boxLeft, boxRight, boxTop, boxBottom);
				referenceTotal.blue += referenceBits[2].query(boxLeft, boxRight, boxTop, boxBottom);
				canvasTotal.red += canvasBits[0].query(boxLeft, boxRight, boxTop, boxBottom);
				canvasTotal.green += canvasBits[1].query(boxLeft, boxRight, boxTop, boxBottom);
				canvasTotal.blue += canvasBits[2].query(boxLeft, boxRight, boxTop, boxBottom);
			} else {
				for (int y = boxTop; y <= boxBottom; y++) {
					for (int x = boxLeft; x <= boxRight; x++) {
						Color c = reference.getPixelColor(x, y);
						referenceTotal.red += c.red;
						referenceTotal.green += c.green;
						referenceTotal.blue += c.blue;
						c = canvas.getPixelColor(x, y);
						canvasTotal.red += c.red;
						canvasTotal.green += c.green;
						canvasTotal.blue += c.blue;
					}
				}
			}

			// number of squares in the rectangle
			int area = (boxRight - boxLeft + 1) * (boxBottom - boxTop + 1);

			Color referenceAverage(referenceTotal);
			referenceAverage.red /= area;
			referenceAverage.green /= area;
			referenceAverage.blue /= area;

			Color canvasAverage(canvasTotal);
			canvasAverage.red /= area;
			canvasAverage.green /= area;
			canvasAverage.blue /= area;

			double oldCost = params.costFunc(referenceAverage, canvasAverage);

			// calculate what the average color in the canvas would be if we drew the string here
			// TODO: support string thickness
			for (int x = x1; x <= x2; x++) {
				int y = round(m * (x - start.x) + start.y);
				Color c = canvas.getPixelColor(x, y);
				canvasTotal.red -= c.red;
				canvasTotal.green -= c.green;
				canvasTotal.blue -= c.blue;
				canvasTotal.red += params.stringColor.red;
				canvasTotal.green += params.stringColor.green;
				canvasTotal.blue += params.stringColor.blue;
			}

			Color stringAverage(canvasTotal);
			stringAverage.red /= area;
			stringAverage.green /= area;
			stringAverage.blue /= area;

			double newCost = params.costFunc(referenceAverage, stringAverage);

			totalImprovement += oldCost - newCost;
		}
	} else {
		// top to bottom
		Point start = (pos1.y < pos2.y) ? pos1 : pos2;
		Point end = (pos1.y < pos2.y) ? pos2 : pos1;

		// reciprocal of slope of line from start to end
		double mInv = (double)(end.x - start.x) / (end.y - start.y);
		int width = reference.getWidth();
		int height = reference.getHeight();

		// visit each rectangle
		for (int y1 = start.y, y2 = 0; y1 < end.y; y1 = y2, numVisited++) {
			int x1 = round(mInv * (y1 - start.y) + start.x);
			y2 = min(y1 + params.rectSize, end.y);
			int x2 = round(mInv * (y2 - start.y) + start.x);
			int midX = (x1 + x2) / 2;
			int midY = (y1 + y2) / 2;
			int boxLeft = max(midX - params.rectSize / 2, 0);
			int boxRight = min(midX + params.rectSize / 2, width - 1);
			int boxTop = max(midY - params.rectSize / 2, 0);
			int boxBottom = min(midY + params.rectSize / 2, height - 1);

			// calculate average color in this rectangle in the reference and canvas images
			Color referenceTotal(0), canvasTotal(0);
			
			if (useBit) {
				referenceTotal.red += referenceBits[0].query(boxLeft, boxRight, boxTop, boxBottom);
				referenceTotal.green += referenceBits[1].query(boxLeft, boxRight, boxTop, boxBottom);
				referenceTotal.blue += referenceBits[2].query(boxLeft, boxRight, boxTop, boxBottom);
				canvasTotal.red += canvasBits[0].query(boxLeft, boxRight, boxTop, boxBottom);
				canvasTotal.green += canvasBits[1].query(boxLeft, boxRight, boxTop, boxBottom);
				canvasTotal.blue += canvasBits[2].query(boxLeft, boxRight, boxTop, boxBottom);
			} else {
				for (int y = boxTop; y <= boxBottom; y++) {
					for (int x = boxLeft; x <= boxRight; x++) {
						Color c = reference.getPixelColor(x, y);
						referenceTotal.red += c.red;
						referenceTotal.green += c.green;
						referenceTotal.blue += c.blue;
						c = canvas.getPixelColor(x, y);
						canvasTotal.red += c.red;
						canvasTotal.green += c.green;
						canvasTotal.blue += c.blue;
					}
				}
			}

			// number of squares in the rectangle
			int area = (boxRight - boxLeft + 1) * (boxBottom - boxTop + 1);

			Color referenceAverage(referenceTotal);
			referenceAverage.red /= area;
			referenceAverage.green /= area;
			referenceAverage.blue /= area;

			Color canvasAverage(canvasTotal);
			canvasAverage.red /= area;
			canvasAverage.green /= area;
			canvasAverage.blue /= area;

			double oldCost = params.costFunc(referenceAverage, canvasAverage);

			// calculate what the average color in the canvas would be if we drew the string here
			// TODO: support string thickness
			for (int y = y1; y <= y2; y++) {
				int x = round(mInv * (y - start.y) + start.x);
				Color c = canvas.getPixelColor(x, y);
				canvasTotal.red -= c.red;
				canvasTotal.green -= c.green;
				canvasTotal.blue -= c.blue;
				canvasTotal.red += params.stringColor.red;
				canvasTotal.green += params.stringColor.green;
				canvasTotal.blue += params.stringColor.blue;
			}

			Color stringAverage(canvasTotal);
			stringAverage.red /= area;
			stringAverage.green /= area;
			stringAverage.blue /= area;

			double newCost = params.costFunc(referenceAverage, stringAverage);

			totalImprovement += oldCost - newCost;
		}
	}

	return totalImprovement / numVisited;
	//return totalImprovement;
}

// The canvas image is the output of the string art drawing. However, we want
// to be able to draw multiple times on the same canvas with different colors.
// This is why the canvas image is taken as a parameter.
void draw(const Image& reference, Image& canvas, const StringArtParams& params) {
	int currPegNum = 0, prevPegNum = -1;
	// count the number of consecutive iterations with zero max improvement
	int countZero = 0;
	for (int iter = 0; iter < params.numIters; iter++) {

		// make the bits
		unsigned width = reference.getWidth();
		unsigned height = reference.getHeight();
		array<Bit<double>, 3> referenceBits {{{width, height}, {width, height}, {width, height}}};
		array<Bit<double>, 3> canvasBits {{{width, height}, {width, height}, {width, height}}};
		for (unsigned y = 0; y < height; y++) {
			for (unsigned x = 0; x < width; x++) {
				Color c = reference.getPixelColor(x, y);
				referenceBits[0].update(x, y, c.red);
				referenceBits[1].update(x, y, c.green);
				referenceBits[2].update(x, y, c.blue);
				c = canvas.getPixelColor(x, y);
				canvasBits[0].update(x, y, c.red);
				canvasBits[1].update(x, y, c.green);
				canvasBits[2].update(x, y, c.blue);
			}
		}

		Point currPegPos = getPegCoords(currPegNum, params.numPegs, canvas);
		double maxImprovement = -numeric_limits<double>::infinity();
		int bestPegNum = -1;

		#pragma omp parallel for num_threads(4)
		for (int nextPegNum = 0; nextPegNum < params.numPegs; nextPegNum++) {
			if (nextPegNum == prevPegNum || nextPegNum == currPegNum) continue;

			// TODO decide whether to keep this or not
			if (abs(nextPegNum - prevPegNum) < 10) continue;

			Point nextPegPos = getPegCoords(nextPegNum, params.numPegs, canvas);
			double improvement = calcImprovement(reference, referenceBits, canvas, canvasBits, currPegPos, nextPegPos, params);
			if (improvement > maxImprovement) {
				#pragma omp atomic write
				maxImprovement = improvement;
				#pragma omp atomic write
				bestPegNum = nextPegNum;
			}
		}

		if (iter % 100 == 0) {
			cout << "done " << iter << endl;
			cout << "max improvement " << maxImprovement << endl;
			canvas.write(params.outputImageFilename);
		}

		Point bestPegPos = getPegCoords(bestPegNum, params.numPegs, canvas);

		// draw the line
		bool leftToRight = abs(currPegPos.x - bestPegPos.x) > abs(currPegPos.y - bestPegPos.y);
		if (leftToRight) {
			Point start = (currPegPos.x < bestPegPos.x) ? currPegPos : bestPegPos;
			Point end = (currPegPos.x < bestPegPos.x) ? bestPegPos : currPegPos;

			// slope of line from start to end
			double m = (double)(end.y - start.y) / (end.x - start.x);

			// draw the line to the canvas
			for (int x = start.x; x <= end.x; x++) {
				int y = round(m * (x - start.x) + start.y);
				Color orig = canvas.getPixelColor(x, y);
				canvas.setPixelColor(x, y, params.stringColor);
				canvasBits[0].update(x, y, params.stringColor.red - orig.red);
				canvasBits[1].update(x, y, params.stringColor.green - orig.green);
				canvasBits[2].update(x, y, params.stringColor.blue - orig.blue);
			}

		} else {
			// top to bottom
			Point start = (currPegPos.y < bestPegPos.y) ? currPegPos : bestPegPos;
			Point end = (currPegPos.y < bestPegPos.y) ? bestPegPos : currPegPos;

			// reciprocal of slope of line from start to end
			double mInv = (double)(end.x - start.x) / (end.y - start.y);

			// draw the line to the canvas
			for (int y = start.y; y <= end.y; y++) {
				int x = round(mInv * (y - start.y) + start.x);
				Color orig = canvas.getPixelColor(x, y);
				canvas.setPixelColor(x, y, params.stringColor);
				canvasBits[0].update(x, y, params.stringColor.red - orig.red);
				canvasBits[1].update(x, y, params.stringColor.green - orig.green);
				canvasBits[2].update(x, y, params.stringColor.blue - orig.blue);
			}
		}

		prevPegNum = currPegNum;
		currPegNum = bestPegNum;

		if (maxImprovement <= 0) {
			countZero++;
			cout << "max improvement 0" << endl;
		} else {
			countZero = 0;
		}

		if (countZero == 100) {
			cout << "Stopping early due to " << countZero << " consecutive iterations with no improvement" << endl;
			break;
		}
	}

}

void makeStringArt(const StringArtParams& params) {
	params.validate();

	Image img {params.inputImageFilename};
	Image output {params.backgroundColor, img.getWidth(), img.getHeight()};

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
}

