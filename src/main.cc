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

Color getColor(string color) {
	if (color[0] == '#') {
		// hex code
		int red =  (color[1] - '0') * 16 + (color[2] - '0');
		int green =  (color[3] - '0') * 16 + (color[4] - '0');
		int blue =  (color[5] - '0') * 16 + (color[6] - '0');
		return Color{red / 256.0, green / 256.0, blue / 256.0}; 
	} else {
		return Color{stod(color.substr(1))};
	}
}

void setOutputFilename(StringArtParams &params, string name) { params.outputImageFilename = name; }
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
	{{"--grayscale-input", "-g"},		setGrayscaleInput,		"convert the input image to grayscale"},
	{{"--rgb-output", "-r"},				setRGBOutput,					"use red, green, and blue strings to generate a color image"},
	{{"--string-color", "-s"},			setStringColor,				"set the color of the lines used to draw the image"},
	{{"--background-color", "-b"},	setBackgroundColor,		"set the background color"},
	{{"--num-pegs", "-p"},					setNumPegs,						"set the number of pegs around the circle"},
	{{"--num-iterations", "-i"},		setNumIters,					"set the number of lines to be drawn"},
	{{"--rect-size", "-r"},					setRectSize,					"set the resolution, in pixels, to use when calculating the best lines to draw"},
	{{"--thickness-func", "-t"},		setThicknessFunc,			"set the function that determines the thickness of the string"},
	{{"--cost-func", "-c"},					setCostFunc,					"set the function that determines the perceived difference between colors"},
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

	/*
	//params.inputImageFilename = "images/new-face-1000.jpg";
	params.inputImageFilename = "images/test-1000.png";
	params.stringColor = Color{0.1};
	params.backgroundColor = Color{0.9};
	params.numPegs = 400;
	params.numIters = 20000;
	params.rectSize = 10;
	//params.thicknessFunc = flatThickness;
	//params.costFunc = absDistanceCost;
	*/

	makeStringArt(params);

	return 0;
}