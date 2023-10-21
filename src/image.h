#ifndef IMAGE_H
#define IMAGE_H

#include <string>

#include <Magick++.h>

#include "color.h"

class Image {
public:
	Image(const std::string& filename);
	Image(Color c, size_t width, size_t height);

	void display() const;

	void convertToGrayscale();
	void write(std::string filename);

	unsigned getWidth() const;
	unsigned getHeight() const;

	// 0-indexed
	Color getPixelColor(unsigned x, unsigned y) const;
	void setPixelColor(unsigned x, unsigned y, const Color &c);

	using Sample = Magick::Quantum;
	struct Pixel {
		Sample red, green, blue, alpha;
	};
private:
	Magick::Image img;
	size_t width, height;

	Pixel *pixels;
	Pixel& get(size_t x, size_t y);
	const Pixel& get(size_t x, size_t y) const;
	Sample maxSample() const;
};

std::ostream& operator<<(std::ostream& os, const Image::Pixel &p);

#endif
