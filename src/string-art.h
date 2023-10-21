#ifndef STRING_ART_H
#define STRING_ART_H

#include <string>

#include "color.h"

typedef double (*ThicknessFunc)(double distance, unsigned sideLength);
typedef double (*CostFunc)(const Color& c1, const Color& c2);

// sample thickness functions
double flatThickness(double distance, unsigned sideLength);
double trapezoidThickness(double distance, unsigned sideLength);

// sample cost functions
double absDistanceCost(const Color& p1, const Color& p2);
double euclideanDistanceCost(const Color& p1, const Color& p2);

/* TODO? 
 *	- transparent background
 *	- crop image to circle
 */
class StringArtParams {
public:
	std::string inputImageFilename;
	std::string outputImageFilename {"tmp/string-art.png"};
	// convert the input image to grayscale
	bool grayscaleInput {true};
	// do three drawing passes, one with each of a 
	// red, green, and blue colored string.
	bool rgbOutput {false};
	Color stringColor {BLACK};
	Color backgroundColor {WHITE};
	int numPegs {200}; 
	int numIters {10000};
	int rectSize {10};
	ThicknessFunc thicknessFunc {trapezoidThickness};
	CostFunc costFunc {euclideanDistanceCost};

	// throws errors if any parameters are invalid
	void validate() const;
};

void makeStringArt(const StringArtParams& params);

#endif
