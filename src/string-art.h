#ifndef STRING_ART_H
#define STRING_ART_H

#include <string>

#include "color.h"

/* TODO? 
 *	- transparent background
 *	- crop image to circle
 */
class StringArtParams {
public:
	std::string inputImageFilename;
	std::string outputFilename {"tmp/string-art.png"};
	bool rgb {false};
	short lineWeight {20}; // virtual line weight
	short stringWeight {20}; // actual line weight
	Color stringColor {BLACK};
	Color backgroundColor {WHITE};
	int numPegs {200}; 
	int numIters {10000};
	bool stopEarly {false};
	bool video {false};
	int numStringsPerFrame {100};
	int fps {10};

	// For example, 10 means any line must be between two pegs that are 10% of the circle away from each other
	int minDist{0};

	// throws errors if any parameters are invalid
	void validate() const;
};

void makeStringArt(StringArtParams params);

#endif
