#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <cmath>

template<typename T>
double calcMagnitude(T *v, int len) {
	double mag = 0;
	for (int i = 0; i < len; i++) {
		mag += v[i] * v[i];
	}
	return sqrt(mag);
}

#endif
