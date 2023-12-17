#ifndef STRING_ART_H
#define STRING_ART_H

#include <string>

#include "color.h"

typedef double (*CostFunc)(const Color& c1, const Color& c2);

// sample cost functions
double absDistanceCost(const Color& p1, const Color& p2);

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
	short lineWeight {20};
	Color stringColor {20};
	Color backgroundColor {WHITE};
	int numPegs {200}; 
	int numIters {10000};

	// 10 means any line must be between two pegs that are 10% of the circle away from each other
	int minDist{10};

	CostFunc costFunc {absDistanceCost};

	// throws errors if any parameters are invalid
	void validate() const;
};

void makeStringArt(const StringArtParams& params);

#endif
