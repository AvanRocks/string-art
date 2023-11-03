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
	return out << "(" << c.red << ", " << c.green << ", " << c.blue << ")";
}

Color& Color::operator+=(const Color &other) {
	red += other.red;
	green += other.green;
	blue += other.blue;
	return *this;
}

Color operator+(const Color &lhs, const Color &rhs) {
	Color result = lhs;
	result += rhs;
	return result;
}

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

Color& Color::operator*=(const double scale) {
	red *= scale;
	green *= scale;
	blue *= scale;
	return *this;
}

Color operator*(const Color &lhs, const double scale) {
	Color result = lhs;
	result *= scale;
	return result;
}

Color operator*(const double scale, const Color &rhs) {
	Color result = rhs;
	result *= scale;
	return result;
}

double operator*(const Color &lhs, const Color &rhs) {
	return lhs.red * rhs.red + lhs.green * rhs.green + lhs.blue * rhs.blue;
}

Color& Color::operator/=(const double scale) {
	red /= scale;
	green /= scale;
	blue /= scale;
	return *this;
}

Color operator/(const Color &lhs, const double scale) {
	Color result = lhs;
	result /= scale;
	return result;
}
