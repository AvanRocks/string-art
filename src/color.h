#ifndef COLOR_H
#define COLOR_H

#include <iostream>

#include "cuda-fix.h"

struct Color {
	double red, green, blue;
	__host__ __device__
	Color(double, double, double);
	__host__ __device__
	Color(double);
	__host__ __device__
	Color() = default; // needed for some CUDA warnings to go away
	// interpolate between c1 and c2. t = 0 return c1, t = 1 returns c2.
	static Color interp(const Color &c1, const Color &c2, double t);

	__host__ __device__
	Color& operator+=(const Color &other);
	__host__ __device__
	Color& operator-=(const Color &other);
	__host__ __device__
	Color& operator*=(const double scale);
	__host__ __device__
	Color& operator/=(const double scale);
};
__host__ __device__
bool operator==(const Color &c1, const Color &c2);
std::ostream &operator<<(std::ostream &out, const Color &c);
__host__ __device__
Color operator+(const Color &lhs, const Color &rhs);
__host__ __device__
Color operator-(const Color &lhs, const Color &rhs);
__host__ __device__
Color operator*(const Color &lhs, const double scale);
__host__ __device__
Color operator*(const double scale, const Color &rhs);
__host__ __device__
Color operator/(const Color &lhs, const double scale);
// dot product
__host__ __device__
double operator*(const Color &lhs, const Color &rhs);
extern Color WHITE;
extern Color BLACK;

#endif
