#include <format>
#include <iostream>
#include <stdexcept>
#include <vector>

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
	this->img.alpha(true);
	this->pixels = reinterpret_cast<Pixel*>(img.getPixels(0, 0, img.columns(), img.rows()));
}

Image::Image(Color c, size_t width, size_t height) 
	: img{
				Magick::Geometry{width, height}, 
				Magick::Color(
											c.red * this->maxSample(),
											c.green * this->maxSample(),
											c.blue * this->maxSample(),
											this->maxSample()
										 )
			 },
		width{width},
		height{height}
{
	this->img.alpha(true);
	this->pixels = reinterpret_cast<Pixel*>(img.getPixels(0, 0, img.columns(), img.rows()));
}

ostream& operator<<(ostream& os, const Image::Pixel &p) {
	return os << format("red: {}, green: {}, blue: {}, alpha: {}", p.red, p.green, p.blue, p.alpha);
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
	/*
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			Color c = this->getPixelColor(x, y);
			//cout << format("({}, {}, {})", c.red, c.green, c.blue) << endl;
			if (c == WHITE) {
				cout << '.';
			} else if (c == BLACK) {
				cout << '#';
			} else {
				cout << 'x';
			}
		}
		cout << endl;
	}
	return;
	*/

	this->img.syncPixels();
	Magick::Image copy {img};
	copy.write(filename);
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
		throw invalid_argument(format("Attempt to access pixel at invalid coordinates (x: {}, y: {}). Image dimensions are width: {}, height: {}.", x, y, this->width, this->height));
	}
	return pixels[this->width * y + x];
}

Sample Image::maxSample() const {
	return QRange;
}

Color Image::getPixelColor(unsigned x, unsigned y) const {
	const Pixel& p = get(x, y);
	Color c { static_cast<double>(p.red) / this->maxSample(), 
						static_cast<double>(p.green) / this->maxSample(), 
						static_cast<double>(p.blue) / this->maxSample() };
	return c;
}

void Image::setPixelColor(unsigned x, unsigned y, const Color &c) {
	Pixel& p = get(x, y);
	p.red = static_cast<Sample>(c.red * this->maxSample());
	p.green = static_cast<Sample>(c.green * this->maxSample());
	p.blue = static_cast<Sample>(c.blue * this->maxSample());
	//cout << format("Color: ({}, {}, {})", p.red, p.green, p.blue) << endl;
}
