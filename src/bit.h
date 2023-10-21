#ifndef BIT_H
#define BIT_H

#include <vector>

template <typename T>
class Bit {
public:
	Bit(unsigned width, unsigned height) 
		: width{width}
		, height{height}
		, bit{height + 1, std::vector<T>(width + 1)}
		{}
	void update(int x, int y, T val) {
		// increment because the bit is 1-indexed
		x++; y++;
		for (; x <= width; x += x&-x) {
			for (int i = y; i <= height; i += i&-i) {
				bit[i][x] += val;
			}
		}
	}
	T query(int left, int right, int top, int bottom) const {
		// increment because the bit is 1-indexed
		left++; right++; top++; bottom++;
		return query(right, bottom) - query(left - 1, bottom) - query(right, top - 1) + query(left - 1 , top - 1);
	}
private:
	T query(int x, int y) const {
		T sum{};
		for (; x; x -= x&-x) {
			for (int i = y; i; i -= i&-i) {
				/*
				T val;
				#pragma omp atomic read
				val = bit[i][x];

				sum += val;
				*/
				sum += bit[i][x];
			}
		}
		return sum;
	}
	unsigned width, height;
	std::vector<std::vector<T>> bit;
};

#endif
