#include "stopwatch.h"

namespace stopwatch {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	std::chrono::milliseconds milli;
}
