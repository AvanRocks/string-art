#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

#include "image.h"
#include "q-range.h"
using Sample = Magick::Quantum;

using namespace std;

/*
 * See:
 *	- https://stackoverflow.com/questions/7678511/getting-pixel-color-with-magick
 */
Image::Image(const string &filename) 
	: img{filename},
		width{img.columns()},
		height{img.rows()}
{
	// for ImageMagick 7
	//this->img.alpha(true);

	this->pixels = reinterpret_cast<Pixel*>(img.getPixels(0, 0, img.columns(), img.rows()));
}

Image::Image(Color c, size_t width, size_t height) 
	: img{
				Magick::Geometry{width, height}, 
				Magick::Color(
											/*
											c.red * this->maxSample(),
											c.green * this->maxSample(),
											c.blue * this->maxSample(),
											*/
											c.red / 255.0 * this->maxSample(),
											c.green / 255.0 * this->maxSample(),
											c.blue / 255.0 * this->maxSample(),

											// For Imagemagick 7
											//this->maxSample()

											// For Imagemagick 6
											0
										 )
			 },
		width{width},
		height{height}
{
	// for ImageMagick 7
	//this->img.alpha(true);

	this->pixels = reinterpret_cast<Pixel*>(img.getPixels(0, 0, img.columns(), img.rows()));
}

ostream& operator<<(ostream& os, const Image::Pixel &p) {
	return os << "red: " << p.red << ", green: " << p.green << ", blue: " << p.blue << ", alpha: " << p.alpha;
}

void Image::display() const { 
	Magick::Image copy {img};
	copy.display(); 
}

unsigned Image::getWidth() const { return width; }
unsigned Image::getHeight() const { return height; }

void Image::convertToGrayscale() {
	for (size_t y = 0; y < this->height; y++) {
		for (size_t x = 0; x < this->width; x++) {
			Pixel &p = get(x, y);
			Sample avg = (p.red + p.green + p.blue) / 3;
			p.red = avg;
			p.green = avg;
			p.blue = avg;
		}
	}
}

void Image::write(std::string filename) {
	this->img.syncPixels();
	Magick::Image copy {img};
	try {
		copy.write(filename);
	} catch (Magick::Warning &warning) {
		// ignore
		// See: https://github.com/ImageMagick/ImageMagick/discussions/6292
	}
}

void Image::writeToPipe(std::FILE *pipe) {
	int width = this->getWidth();
	int height = this->getHeight();
	int sample_depth = 3;
	int size = width * height * sample_depth;
	char *frameData = new char[size];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int idx = (y * width + x) * sample_depth;
			// use bgr24 pixel format
			Color c {this->getPixelColor(x, y)};
			frameData[idx] = (unsigned char) c.blue;
			frameData[idx + 1] = (unsigned char) c.green;
			frameData[idx + 2] = (unsigned char) c.red;
		}
	}
	fwrite(frameData, 1, size, pipe);
}

// 0-indexed
Image::Pixel& Image::get(size_t x, size_t y) {
	return const_cast<Pixel&>(const_cast<const Image*>(this)->get(x, y));
}

/*
 * See:
 *	- https://stackoverflow.com/questions/856542/elegant-solution-to-duplicate-const-and-non-const-getters
 */
const Image::Pixel& Image::get(size_t x, size_t y) const {
	if (!(0 <= x && x < this->width && 0 <= y && y < this->height)) {
		throw invalid_argument(
				"Attempt to access pixel at invalid coordinates (x: " 
				+ to_string(x) 
				+ ", y: " 
				+ to_string(y) 
				+ "). Image dimensions are width: " 
				+ to_string(this->width) 
				+ ", height: " 
				+ to_string(this->height) 
				+ ".");
	}
	return pixels[this->width * y + x];
}

Sample Image::maxSample() const {
	return QRange;
}

Color Image::getPixelColor(unsigned x, unsigned y) const {
	const Pixel& p = get(x, y);
	/*
	Color c { static_cast<double>(p.red) / this->maxSample(), 
						static_cast<double>(p.green) / this->maxSample(), 
						static_cast<double>(p.blue) / this->maxSample() };
	return c;
	*/
	Color c { (short)(static_cast<double>(p.red) / this->maxSample() * 255), 
						(short)(static_cast<double>(p.green) / this->maxSample() * 255),
						(short)(static_cast<double>(p.blue) / this->maxSample() * 255) };
	return c;
	//return ((p.red + p.green + p.blue) / 3.0) * (256.0 / this->maxSample());
}

void Image::setPixelColor(unsigned x, unsigned y, const Color &c) {
	Pixel& p = get(x, y);
	p.red = static_cast<Sample>(c.red / 255.0 * this->maxSample());
	p.green = static_cast<Sample>(c.green / 255.0 * this->maxSample());
	p.blue = static_cast<Sample>(c.blue / 255.0 * this->maxSample());
	/*
	Sample color = (c / 256.0) * this->maxSample();
	p.red = color;
	p.green = color;
	p.blue = color;
	*/
}
