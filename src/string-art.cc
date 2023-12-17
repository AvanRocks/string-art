#include <iostream>
#include <cmath>
#include <set>
#include <queue>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <deque>

#include "string-art.h"
#include "image.h"
#include "stopwatch.h"
using namespace std;

void StringArtParams::validate() const {
	if (this->inputImageFilename.size() == 0) {
		throw logic_error("Input image filename is empty.");
	}
}

double absDistanceCost(const Color& c1, const Color& c2) {
	return abs(c1 - c2);
}

struct Point {
	short x, y;
};

vector<Point> precomputePegCoords(const int numPegs, const Image &img) {
	vector<Point> pegCoords;
	for (int peg = 0; peg < numPegs; peg++) {
		double theta = 2.0 * numbers::pi * peg / numPegs;
		double x = cos(theta);
		double y = sin(theta);

		short width = img.getWidth();
		short height = img.getHeight();

		short imgX = round(((width - 1) / 2.0) * (1 + x));
		short imgY = round(((height - 1) / 2.0) * (1 - y));

		pegCoords.emplace_back(imgX, imgY);
	}
	return pegCoords;
}

vector<vector<vector<Point>>> precomputeLines(const int numPegs, const Image &img, int minDistPegs) {
	startStopwatch("precomputing lines");
	vector<vector<vector<Point>>> lines(numPegs, vector<vector<Point>>(numPegs));
	vector<Point> pegCoords = precomputePegCoords(numPegs, img);
	unsigned long long size = 0;
	for (int startPeg = 0; startPeg < numPegs; startPeg++) {
		for (int endPeg = 0; endPeg < numPegs; endPeg++) {
			if (abs(startPeg - endPeg) <= minDistPegs) continue;
			if (startPeg == endPeg) continue;
			Point pos1 = pegCoords[startPeg];
			Point pos2 = pegCoords[endPeg];

			int lenX = abs(pos1.x - pos2.x);
			int lenY = abs(pos1.y - pos2.y);

			bool leftToRight = lenX > lenY;

			if (leftToRight) {
				Point start = (pos1.x < pos2.x) ? pos1 : pos2;
				Point end = (pos1.x < pos2.x) ? pos2 : pos1;

				// slope of line from start to end
				double m = (double)(end.y - start.y) / (end.x - start.x);

				for (short x = start.x; x < end.x; x++) {
					short y = round(m * (x - start.x) + start.y);
					lines[startPeg][endPeg].push_back({x, y});
					lines[endPeg][startPeg].push_back({x, y});
				}
			} else {
				// top to bottom
				Point start = (pos1.y < pos2.y) ? pos1 : pos2;
				Point end = (pos1.y < pos2.y) ? pos2 : pos1;

				// reciprocal of slope of line from start to end
				double mInv = (double)(end.x - start.x) / (end.y - start.y);

				for (short y = start.y; y < end.y; y++) {
					short x = round(mInv * (y - start.y) + start.x);
					lines[startPeg][endPeg].push_back({x, y});
					lines[endPeg][startPeg].push_back({x, y});
				}
			}
			size += lines[startPeg][endPeg].size();
			size += lines[endPeg][startPeg].size();
		}
	}
	cout << "lines cache size: " << size << endl;
	endStopwatch();
	return lines;
}

// calculate the improvement that would be gained by drawing a line
double calcImprovement(
		const Image &reference,
		const Image &virtualCanvas,
		const vector<vector<vector<Point>>> &lines,
		const int peg1,
		const int peg2,
		const StringArtParams &params
		) 
{

	double totalImprovement = 0;

	for (const Point &p : lines[peg1][peg2]) {
		Color referenceColor = reference.getPixelColor(p.x, p.y);
		Color virtualCanvasColor = virtualCanvas.getPixelColor(p.x, p.y);

		double oldCost = params.costFunc(referenceColor, virtualCanvasColor);

		// color after drawing a line here
		Color stringColor = max(0, virtualCanvasColor -  params.lineWeight);

		double newCost = params.costFunc(referenceColor, stringColor);

		totalImprovement += oldCost - newCost;
	}

	int numPixelsDrawn = lines[peg1][peg2].size();
	return totalImprovement / numPixelsDrawn;
}

// The lines on the virtual canvas are colored as params.lineWeight, while the
// actualCanvas has lines of color params.stringColor
void draw(const Image& reference, Image &virtualCanvas, Image& actualCanvas, const StringArtParams& params) {
	int minDistPegs = params.minDist / 100.0 * params.numPegs;
	vector<vector<vector<Point>>> lines {precomputeLines(params.numPegs, reference, minDistPegs)};

	int currPegNum = 0;
	deque<int> lastPegNums;
	// count the number of consecutive iterations with zero max improvement
	int countZero = 0;
	for (int iter = 0; iter < params.numIters; iter++) {

		double maxImprovement = -numeric_limits<double>::infinity();
		int bestPegNum = -1;

		#pragma omp parallel for
		for (int nextPegNum = 0; nextPegNum < params.numPegs; nextPegNum++) {
			if (abs(nextPegNum - currPegNum) <= minDistPegs) continue;

			if (find(lastPegNums.begin(), lastPegNums.end(), nextPegNum) != lastPegNums.end()) continue;

			// use virtual canvas to calculate improvement
			double improvement = calcImprovement(reference, virtualCanvas, lines, currPegNum, nextPegNum, params);
			if (improvement > maxImprovement) {
				#pragma omp atomic write
				maxImprovement = improvement;
				#pragma omp atomic write
				bestPegNum = nextPegNum;
			}
		}

		if (bestPegNum == -1) {
			throw runtime_error("no pegs available to draw to");
		}

		if (iter % 1000 == 0) {
			cout << "done " << iter << endl;
			cout << "max improvement " << maxImprovement << endl;
			actualCanvas.write(params.outputImageFilename);
		}


		// draw a line with Color params.stringColor onto the actualCanvas
		// and draw a line with Color params.lineWeight onto the virtualCanvas
		for (const Point &p : lines[currPegNum][bestPegNum]) {
			actualCanvas.setPixelColor(p.x, p.y, max(0, actualCanvas.getPixelColor(p.x, p.y) - params.stringColor));
			virtualCanvas.setPixelColor(p.x, p.y, max(0, virtualCanvas.getPixelColor(p.x, p.y) - params.lineWeight));
		}

		currPegNum = bestPegNum;
		lastPegNums.push_back(currPegNum);
		if (lastPegNums.size() > 20) {
			lastPegNums.pop_front();
		}

		if (maxImprovement <= 0) {
			countZero++;
		} else {
			countZero = 0;
		}

		if (countZero == 1000) {
			cout << "Stopping early due to " << countZero << " consecutive iterations with no improvement" << endl;
			break;
		}
	}

}

void makeStringArt(const StringArtParams& params) {
	params.validate();

	Image img {params.inputImageFilename};
	Image virtualCanvas {params.backgroundColor, img.getWidth(), img.getHeight()};
	Image actualCanvas {params.backgroundColor, img.getWidth(), img.getHeight()};

	if (params.grayscaleInput) {
		img.convertToGrayscale();
	}
	img.write("tmp/grayscale.png");

	if (params.rgbOutput) {
		// TODO
		// do three passes, one for each color
		StringArtParams newParams {params};
		/*
		newParams.stringColor = Color{1, 0, 0};
		draw(img, virtualCanvas, actualCanvas, newParams);
		newParams.stringColor = Color{0, 1, 0};
		draw(img, virtualCanvas, actualCanvas, newParams);
		newParams.stringColor = Color{0, 0, 1};
		draw(img, virtualCanvas, actualCanvas, newParams);
		*/
	} else {
		// single pass with specified color
		draw(img, virtualCanvas, actualCanvas, params);
	}

	actualCanvas.write(params.outputImageFilename);
}

