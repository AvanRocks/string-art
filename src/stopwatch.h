#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <chrono>
#include <iostream>

namespace stopwatch {
	extern std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	extern std::chrono::milliseconds milli;
}

#define startStopwatch(desc) \
	{ \
		std::cout << desc << " ... " << std::flush; \
		stopwatch::start = std::chrono::high_resolution_clock::now(); \
	}

#define endStopwatch() \
	{ \
		stopwatch::end = std::chrono::high_resolution_clock::now(); \
		stopwatch::milli = std::chrono::duration_cast<std::chrono::milliseconds>(stopwatch::end - stopwatch::start); \
		std::cout << "DONE (time: " << stopwatch::milli.count() << "ms)" << std::endl; \
	}

#endif
