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

Color getColor(string color) {
	if (color[0] == '#') {
		// hex code
		int red =  getHexDigit(color[1]) * 16 + getHexDigit(color[2]);
		int green =  getHexDigit(color[3]) * 16 + getHexDigit(color[4]);
		int blue =  getHexDigit(color[5]) * 16 + getHexDigit(color[6]);
		//Color c {red / 256.0, green / 256.0, blue / 256.0}; 
		Color c = (red / 256.0 + green / 256.0 + blue / 256.0) / 3.0;
		cout << "color: " << c << endl;
		return c;
	} else {
		return Color{stod(color.substr(1))};
	}
}

void setOutputFilename(StringArtParams &params, string name) { params.outputImageFilename = name; }
void setCacheFilename(StringArtParams &params, string name) { params.matrixCacheFilename = name; }
void setGrayscaleInput(StringArtParams &params) { params.grayscaleInput = true; }
void setRGBOutput(StringArtParams &params) { params.rgbOutput = true; }
void setStringColor(StringArtParams &params, string color) { params.stringColor = getColor(color); }
void setBackgroundColor(StringArtParams &params, string color) { params.backgroundColor = getColor(color); }
void setNumPegs(StringArtParams &params, string numPegs) { params.numPegs = stoi(numPegs); }
void setNumIters(StringArtParams &params, string numIters) { params.numIters = stoi(numIters); }
void setRectSize(StringArtParams &params, string rectSize) { params.rectSize = stoi(rectSize); }

void setThicknessFunc(StringArtParams &params, string thicknessFunc) { 
	if (thicknessFunc == "flat") {
		params.thicknessFunc = flatThickness;
	} else if (thicknessFunc == "trapezoid") {
		params.thicknessFunc = trapezoidThickness;
	} else {
		cout << "Unrecognized thickness function" << endl;
	}
}

void setCostFunc(StringArtParams &params, string costFunc) { 
	if (costFunc == "abs") {
		params.costFunc = absDistanceCost;
	} else if (costFunc == "euclidean") {
		params.costFunc = euclideanDistanceCost;
	} else {
		cout << "Unrecognized cost function" << endl;
	}
}

void usage(char *name);

vector<tuple<vector<string>, variant<NoParamFunc, StringParamFunc, HelpFunc>, string>> cliParams 
{
	{{"--output", "-o"},						setOutputFilename,		"set the output image filename"},
	{{"--cache-file", "-c"},				setCacheFilename,		"set the matrix cache filename"},
	{{"--grayscale-input", "-g"},		setGrayscaleInput,		"convert the input image to grayscale"},
	{{"--rgb-output", "-rgb"},				setRGBOutput,					"use red, green, and blue strings to generate a color image"},
	{{"--string-color", "-s"},			setStringColor,				"set the color of the lines used to draw the image"},
	{{"--background-color", "-b"},	setBackgroundColor,		"set the background color"},
	{{"--num-pegs", "-p"},					setNumPegs,						"set the number of pegs around the circle"},
	{{"--num-iterations", "-i"},		setNumIters,					"set the number of lines to be drawn"},
	{{"--rect-size", "-r"},					setRectSize,					"set the resolution, in pixels, to use when calculating the best lines to draw"},
	{{"--thickness-func", "-t"},		setThicknessFunc,			"set the function that determines the thickness of the string"},
	{{"--cost-func"},					setCostFunc,					"set the function that determines the perceived difference between colors"},
	{{"--help", "-h"},							usage,								"display this help and exit"}
};

void usage(char *name) {
	cout << "Usage: " << name << " [options ...] file" << endl;
	cout << "Generate string art style images." << endl; 
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
		cout << "Please specify an input filename." << endl;
		usage(argv[0]);
		exit(1);
	}

	makeStringArt(params);

	return 0;
}
