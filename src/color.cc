#include "color.h"
using namespace std;

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
	return out << format("({}, {}, {})", c.red, c.green, c.blue);
}
