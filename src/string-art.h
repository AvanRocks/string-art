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
	std::string outputFilename {"string-art.png"};
	bool rgb {false};
	short lineWeight {20}; // virtual line weight
	short stringWeight {20}; // actual line weight
	Color stringColor {BLACK};
	Color backgroundColor {WHITE};
	int numPegs {200}; 
	int numLines {10000};
	bool stopEarly {false};
	bool video {false};

	// minimum arc that a chord must subtend
	int minArc{0};

	// not user-settable
	int numStringsPerFrame {100};
	int fps {10};

	// throws errors if any parameters are invalid
	void validate() const;
};

void makeStringArt(StringArtParams params);

#endif
