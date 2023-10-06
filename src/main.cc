#include <iostream>
#include <Magick++.h>

#include "string-art.h"
#include "image.h"

using namespace std;
//using namespace Magick;

int main() {
	/*
	Image img {"images/house.tiff"}; 
	int cols = img.columns();
	int rows = img.rows();
	string format = img.magick();
	img.write("test.png");

	Image tmp {img};
	tmp.display();
	*/

	/*
	Image orig {"images/house.tiff"}; 
	Image cropped {orig};
	//cropped.alpha(true);
	//cout << cropped.alpha() << '\n';
	//cout << TransparentAlpha << '\n';
	//cout << OpaqueAlpha << '\n';
	cropped.alpha(static_cast<unsigned int>(TransparentAlpha)); // transparent
	//cropped.alpha(static_cast<unsigned int>(OpaqueAlpha / 2));
	cropped.fillColor(Color(0, 0, 0, QuantumRange)); // opaque
	cropped.draw(DrawableCircle(100, 100, 50, 100));

	// Note that display does not support transparency
	//cropped.display();
	
	orig.composite(cropped, 0, 0, CopyAlphaCompositeOp);
	orig.write("images/test.png");
	*/

	StringArtParams params;
	params.inputImageFilename = "images/test-768.png";
	params.stringColor = Color{0.2};
	params.backgroundColor = Color{0.9};
	params.numPegs = 300;
	params.numIters = 10000;
	//params.thicknessFunc = flatThickness;
	//params.costFunc = absDistanceCost;
	makeStringArt(params);

	return 0;
}
