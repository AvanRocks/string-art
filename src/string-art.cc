#include <iostream>
#include <format>
#include <cmath>
#include <set>
#include <queue>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <random>

#include "string-art.h"
#include "image.h"
using namespace std;

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
pair<int, int> getPegCoords(int n, int N, const Image &img) {
	double theta = 2.0 * numbers::pi * n / N;
	double x = cos(theta);
	double y = sin(theta);

	unsigned width = img.getWidth();
	unsigned height = img.getHeight();

	int imgX = round(((width - 1) / 2.0) * (1 + x));
	int imgY = round(((height - 1) / 2.0) * (1 - y));

	return { imgX, imgY };
}

void visit(const Image &reference, queue<pair<int, int>> &q, const set<pair<int, int>> &visited, int x, int y) {
	if (0 <= x && x < reference.getWidth() && 0 <= y && y < reference.getHeight() && !visited.contains({x, y})) {
		q.emplace(x, y);
	}
}

template<typename BfsCallback>
void bfs(const Image &reference, const pair<int, int> &pos1, const pair<int, int> &pos2, BfsCallback callback) {
	//cout << format("starting bfs from ({}, {})", start.first, start.second) << endl;

	// a loose bounding box around pos1 and pos2
	const int EPS = 5;
	int left = min(pos1.first, pos2.first) - EPS;
	int right = max(pos1.first, pos2.first) + EPS;
	int top = min(pos1.second, pos2.second) - EPS;
	int bottom = max(pos1.second, pos2.second) + EPS;

	set<pair<int, int>> visited;
	queue<pair<int, int>> q;
	q.push(pos1);
	while (!q.empty()) {
		pair<int, int> curr = q.front();
		q.pop();
		if (visited.contains(curr)) continue;
		// check if curr is in the loose bounding box defined above
		if (!(left <= curr.first && curr.first <= right && top <= curr.second && curr.second <= bottom)) continue;

		visited.emplace(curr);
		//cout << format("visiting ({}, {})", curr.first, curr.second) << endl;

		if (callback(curr)) {
			visit(reference, q, visited, curr.first - 1, curr.second);
			visit(reference, q, visited, curr.first + 1, curr.second);
			visit(reference, q, visited, curr.first, curr.second - 1);
			visit(reference, q, visited, curr.first, curr.second + 1);
			visit(reference, q, visited, curr.first - 1, curr.second - 1);
			visit(reference, q, visited, curr.first - 1, curr.second + 1);
			visit(reference, q, visited, curr.first + 1, curr.second - 1);
			visit(reference, q, visited, curr.first + 1, curr.second + 1);
		}
	}
}

// return the distance of point r from the line defined by pq
double getDistance(const pair<int, int> &p, const pair<int, int> &q, const pair<int, int> &r) {
	// See: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
	return abs( (q.first - p.first) * (p.second - r.second) - (p.first - r.first) * (q.second - p.second) ) 
							/ (double) sqrt( pow(q.first - p.first, 2) + pow(q.second - p.second, 2) );
}

double calcImprovement(
		const Image &reference, 
		const Image &canvas, 
		const pair<int, int> &pos1, 
		const pair<int, int> &pos2, 
		const StringArtParams &params
		) 
{

	double totalImprovement = 0;
	int numVisited = 0;
	bfs(reference, pos1, pos2, [&] (const pair<int, int> &curr) -> bool {
		// calculate old cost
		Color referencePixelColor = reference.getPixelColor(curr.first, curr.second);
		Color oldPixelColor = canvas.getPixelColor(curr.first, curr.second);
		double oldCost = params.costFunc(referencePixelColor, oldPixelColor);

		double distance = getDistance(pos1, pos2, curr);
		double thickness = params.thicknessFunc(distance, reference.getWidth());
		if (thickness == 0) return false;
		//cout << format("distance: {}, thickness: {}", distance, thickness) << endl;
		Color newPixelColor = Color::interp(oldPixelColor, params.stringColor, thickness);

		double newCost = params.costFunc(referencePixelColor, newPixelColor);

		//cout << format("oldCost: {}, newCost: {}", oldCost, newCost) << endl;

		// TODO: check if totalImprovement overflows
		double improvement = oldCost - newCost;
		totalImprovement += improvement;

		numVisited++;

		return true;
	});

	//cout << format("total improvement: {}", totalImprovement) << endl;
	return totalImprovement / numVisited;
}


random_device rd;
mt19937 rng(rd());

bool randChance(double probSuccess) {
	unsigned randomNum = rng();
	return randomNum < probSuccess * mt19937::max();
}

int randNum(int min, int max) {
	int range = max - min + 1;
	int randomNum = rng() % range;
	return randomNum + min;
}

// The canvas image is the output of the string art drawing. However, we want
// to be able to draw multiple times on the same canvas with different colors.
// This is why the canvas image is taken as a parameter.
void draw(const Image& reference, Image& canvas, const StringArtParams& params) {
	int currPegNum = 0, prevPegNum = -1;
	for (int iter = 0; iter < params.numIters; iter++) {
		pair<int, int> currPegPos = getPegCoords(currPegNum, params.numPegs, canvas);
		double maxImprovement = -numeric_limits<double>::infinity();
		int bestPegNum = -1;

		#pragma omp parallel for
		for (int nextPegNum = 0; nextPegNum < params.numPegs; nextPegNum++) {
			if (nextPegNum == prevPegNum || nextPegNum == currPegNum) continue;
			if (abs(nextPegNum - prevPegNum) < 10) continue;
			pair<int, int> nextPegPos = getPegCoords(nextPegNum, params.numPegs, canvas);
			double improvement = calcImprovement(reference, canvas, currPegPos, nextPegPos, params);
			if (improvement > maxImprovement) {
				#pragma omp atomic write
				maxImprovement = improvement;
				#pragma omp atomic write
				bestPegNum = nextPegNum;
			}
		}

		if (randChance(0.1)) bestPegNum += randNum(-10, 10);

		if (iter % 100 == 0) {
			cout << format("done {}", iter) << endl;
			cout << format("max improvement {}", maxImprovement) << endl;
		}
		//cout << format("curr peg pos ({}, {})", currPegPos.first, currPegPos.second) << endl;
		//cout << format("curr peg num: {}", currPegNum) << endl;

		pair<int, int> bestPegPos = getPegCoords(bestPegNum, params.numPegs, canvas);
		bfs(reference, currPegPos, bestPegPos, [&] (const pair<int, int> &curr) -> bool {
			Color oldPixelColor = canvas.getPixelColor(curr.first, curr.second);

			double distance = getDistance(currPegPos, bestPegPos, curr);
			double thickness = params.thicknessFunc(distance, reference.getWidth());
			if (thickness == 0) return false;
			Color newPixelColor = Color::interp(oldPixelColor, params.stringColor, thickness);

			canvas.setPixelColor(curr.first, curr.second, newPixelColor);

			return true;
		});

		prevPegNum = currPegNum;
		currPegNum = bestPegNum;

		if (maxImprovement < 0) {
			cout << format("Stopped early at iteration number {}", iter) << endl;
			break;
		}
	}

}

void makeStringArt(const StringArtParams& params) {
	/*
	Image img {params.inputImageFilename};
	Image output {WHITE, img.getWidth(), img.getHeight()};

	for (int i = 0; i < params.numPegs; i++) {
		auto pos = getPegCoords(i, params.numPegs, output);
		output.setPixelColor(pos.first, pos.second, BLACK);
	}
	output.write("tmp/test.png");
	*/

	/*
	pair<int, int> p {0, 0};
	pair<int, int> q {1, 100};
	pair<int, int> r {0.5, 51};
	cout << getDistance(p, q, r) << endl;
	return;
	*/

	params.validate();

	Image img {params.inputImageFilename};
	Image output {params.backgroundColor, img.getWidth(), img.getHeight()};

	if (params.grayscaleInput) {
		img.convertToGrayscale();
	}
	img.write("tmp/grayscale.tiff");

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

