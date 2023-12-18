#include "color.h"
#include <cmath>
using namespace std;

Color WHITE {255, 255, 255};
Color BLACK {0, 0, 0};

Color::Color(short red, short green, short blue) : red{red}, green{green}, blue{blue} {}
Color::Color(short x) : Color{x, x, x} {}

Color& Color::operator-=(const Color &other) {
	red -= other.red;
	green -= other.green;
	blue -= other.blue;
	return *this;
}

Color operator-(const Color &lhs, const Color &rhs) {
	Color result = lhs;
	result -= rhs;
	return result;
}

/*
Color& Color::clamp() {
	const short min = 0, max = 255;
	red = std::clamp(red, min, max);
	green = std::clamp(green, min, max);
	blue = std::clamp(blue, min, max);
	return *this;
}
*/

/*
Color& Color::invert() {
	red = 255 - red;
	green = 255 - green;
	blue = 255 - blue;
	return *this;
}
*/

Color& Color::towards(const Color &other, short amount) {
	double deltaRed = other.red - this->red;
	double deltaGreen = other.green - this->green;
	double deltaBlue = other.blue - this->blue;
	double magnitude = sqrt(deltaRed * deltaRed + deltaGreen * deltaGreen + deltaBlue * deltaBlue);
	if (magnitude > amount) {
		deltaRed /= magnitude;
		deltaGreen /= magnitude;
		deltaBlue /= magnitude;
		this->red += (short)(deltaRed * amount);
		this->green += (short)(deltaGreen * amount);
		this->blue += (short)(deltaBlue * amount);
	} else {
		// don't overshoot
		this->red = other.red;
		this->green = other.green;
		this->blue = other.blue;
	}
	return *this;
}

/*
Color WHITE {256};
Color BLACK {0};
*/
/*
Color WHITE{1, 1, 1};
Color BLACK{0, 0, 0};

Color::Color(double red, double green, double blue) : red{red}, green{green}, blue{blue} {}
Color::Color(double d) : Color{d, d, d} {}

Color Color::interp(const Color &c1, const Color &c2, double t) {
	return {
		c1.red * (1 - t) + c2.red * t,
		c1.green * (1 - t) + c2.green * t,
		c1.blue * (1 - t) + c2.blue * t
	};
}

bool operator==(const Color &c1, const Color &c2) {
	return c1.red == c2.red 
				&& c1.green == c2.green
				&& c1.blue == c2.blue;
}

ostream &operator<<(ostream &out, const Color &c) {
	return out << "(" << c.red << ", " << c.green << ", " << c.blue << ")";
}
*/
