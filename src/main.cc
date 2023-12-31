#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <variant>
#include <stdexcept>
#include <iterator>
#include <iomanip>
#include <numeric>

#include "string-art.h"

using namespace std;

typedef void (*StringParamFunc)(StringArtParams &params, string arg);
typedef void (*NoParamFunc)(StringArtParams &params);
typedef void (*HelpFunc)(char *name);

int getHexDigit(char c) {
	if ('0' <= c && c <= '9') {
		return c - '0';
	} else if ('A' <= c && c <= 'F') {
		return (c - 'A') + 10;
	} else if ('a' <= c && c <= 'f') {
		return (c - 'a') + 10;
	} else {
		throw logic_error("invalid hex code");
	}
}

bool inRange(int val, int min, int max) {
	return min <= val && val <= max;
}

Color getColor(string color) {
	if (color[0] == '#') {
		// hex code
		int red =  getHexDigit(color[1]) * 16 + getHexDigit(color[2]);
		int green =  getHexDigit(color[3]) * 16 + getHexDigit(color[4]);
		int blue =  getHexDigit(color[5]) * 16 + getHexDigit(color[6]);
		if (!(inRange(red, 0, 255) && inRange(green, 0, 255) && inRange(blue, 0, 255))) {
			throw invalid_argument("invalid color hex code: " + color);
		}
		return Color{(short)red, (short)green, (short)blue};
	} else {
		int value = stoi(color);
		if (!inRange(value, 0, 255)) {
			throw invalid_argument("invalid color: " + color);
		}
		return Color{(short)stoi(color)};
	}
}

void setOutputFilename(StringArtParams &params, string name) { params.outputFilename = name; }
void setRGB(StringArtParams &params) { params.rgb = true; }
void setLineWeight(StringArtParams &params, string weight) { params.lineWeight = stoi(weight); }
void setStringWeight(StringArtParams &params, string weight) { params.stringWeight = stoi(weight); }
void setStringColor(StringArtParams &params, string color) { params.stringColor = getColor(color); }
void setBackgroundColor(StringArtParams &params, string color) { params.backgroundColor = getColor(color); }
void setNumPegs(StringArtParams &params, string numPegs) { params.numPegs = stoi(numPegs); }
void setNumLines(StringArtParams &params, string numLines) { params.numLines = stoi(numLines); }
void setMinArc(StringArtParams &params, string minArc) { params.minArc = stoi(minArc); }
void setStopEarly(StringArtParams &params) { params.stopEarly = true; }
void setVideo(StringArtParams &params) { params.video = true; }

void usage(char *name);

vector<tuple<vector<string>, variant<NoParamFunc, StringParamFunc, HelpFunc>, string>> cliParams 
{
	{{"--output",            "-o"},     setOutputFilename,     "set the output filename (default \"string-art.png\")"},
	{{"--num-pegs",          "-p"},     setNumPegs,            "set the number of pegs around the circle (default 200)"},
	{{"--num-lines",         "-l"},     setNumLines,           "set the number of lines to be drawn (default 10,000)"},
	{{"--string-color",      "-s"},     setStringColor,        "set the color of the lines used to draw the image (default black)"},
	{{"--background-color",  "-b"},     setBackgroundColor,    "set the background color (default white)"},
	{{"--video",             "-v"},     setVideo,              "output a video of the drawing process (default false)"},
	{{"--rgb",               "-rgb"},   setRGB,                "use red, green, and blue strings to generate a color image (default false)"},
	{{"--stop-early",        "-stop"},  setStopEarly,          "stop the generation after 1000 lines with no improvement (default false)"},
	{{"--line-weight",       "-lw"},    setLineWeight,         "set how dark the program thinks the virtual lines are (0-255) (default 20)"},
	{{"--string-weight",     "-sw"},    setStringWeight,       "set how dark the program thinks the string is (0-255) (default 20)"},
	{{"--min-arc",           "-m"},     setMinArc,             "set the minimum arc that chords must subtend, in degrees (default 0)"},
	{{"--help",              "-h"},     usage,                 "display this help and exit"}
};

void usage(char *name) {
	cout << "Usage: " << name << " [options ...] file" << endl;
	cout << "Generate string art style images" << endl; 
	cout << endl;
	cout << "Options:" << endl;
	for (const auto &[paramList, func, helpString] : cliParams) {
		cout << "  ";
		copy(paramList.begin(), paramList.end(), ostream_iterator<string>(cout, ", "));
		const int WIDTH = 22;
		int len = accumulate(paramList.begin(), paramList.end(), 0, [](int sum, const string &s) { return sum + s.size(); });
		for (int i = 0; i < max(0, WIDTH - len); i++) {
			cout << " ";
		}
		cout << helpString << endl;
	}
}

int main(int argc, char **argv) {
	StringArtParams params;

	bool inputFileNameFound = false;
	for (int i = 1; i < argc; i++) {

		if (argv[i][0] == '-') {
			bool paramFound = false;
			for (const auto &[options, func, helpString] : cliParams) {
				auto it = find(options.begin(), options.end(), argv[i]);
				if (it != options.end()) {
					if (holds_alternative<StringParamFunc>(func)) {
						if (i + 1 < argc) {
							get<StringParamFunc>(func)(params, argv[i+1]);
							// skip next cli argument
							i++;
						} else {
							cout << "Option requires an argument: " << argv[i] << endl;
							cout << "Try " << argv[0] << " --help for more information." << endl;
						}
					} else if (holds_alternative<NoParamFunc>(func)){
						get<NoParamFunc>(func)(params);
					} else if (holds_alternative<HelpFunc>(func)) {
						get<HelpFunc>(func)(argv[0]);
						exit(0);
					}

					paramFound = true;
					break;
				}
			}

			if (!paramFound) {
				cout << "Unrecognized option: " << argv[i] << endl;
				usage(argv[0]);
				exit(1);
			}
		} else if (inputFileNameFound == false) {
			// first string that is not an option, assume it is an input filename
			params.inputImageFilename = argv[i];
			inputFileNameFound = true;
		} else {
			// error
			usage(argv[0]);
			exit(1);
		}
	}

	if (inputFileNameFound == false) {
		cout << "Please specify an input filename" << endl;
		usage(argv[0]);
		exit(1);
	}

	try {
		makeStringArt(params);
	} catch (const exception &e) {
		cout << "Error: " << e.what() << endl;
	}

	return 0;
}
