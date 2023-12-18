#ifndef COLOR_H
#define COLOR_H

#include <iostream>

struct Color {
	short red, green, blue;
	Color(short, short, short);
	Color(short);
	Color& operator-=(const Color &other);
	//Color& clamp();
	//Color& invert();
	Color& towards(const Color &other, short amount);
};
Color operator-(const Color &lhs, const Color &rhs);

extern Color WHITE;
extern Color BLACK;

/*
using Color = short;
extern Color WHITE;
extern Color BLACK;
*/

/*
struct Color {
	double red, green, blue;
	Color(double, double, double);
	Color(double);
	// interpolate between c1 and c2. t = 0 return c1, t = 1 returns c2.
	static Color interp(const Color &c1, const Color &c2, double t);
};
bool operator==(const Color &c1, const Color &c2);
std::ostream &operator<<(std::ostream &out, const Color &c);
extern Color WHITE;
extern Color BLACK;
*/

#endif
