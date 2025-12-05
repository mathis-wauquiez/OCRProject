/**
 * @file utility.h
 * @brief Utility functions
 * @author Yuchen He <yuchenroy@sjtu.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * You should have received a copy of the GNU General Pulic License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <cassert>
#include "lltree.h"
#include "image.h"
#include "curv.h"
#include "cmdLine.h"
#include "gass.h"
#include "xmtime.h"
#include "lltree.h"
#include "io_png.h"

using namespace std;

unsigned char* readImage(char** argv, size_t* pW, size_t* pH);

void readCmdLine(int& argc, char** argv, float& offset, float& scale, float& global_thresh, int& refine_iteration,
	string& BW_output_path, string& result1, string& result2,
	string& result3, string& result4);

static void smooth(std::vector<Point>& line, double lastScale) {
	std::vector<DPoint> dline;
	for (std::vector<Point>::iterator it = line.begin(); it != line.end(); ++it)
		dline.push_back(DPoint((double)it->x, (double)it->y));

	assert(dline.front() == dline.back());
	gass(dline, 0.0, lastScale);
	line.clear();
	for (std::vector<DPoint>::iterator it = dline.begin(); it != dline.end(); ++it)
		line.push_back(Point((float)it->x, (float)it->y));
}

static void fix_level(LLTree& tree, float offset) {
	for (LLTree::iterator it = tree.begin(PostOrder); it != tree.end(); ++it) {
		float level = it->ll->level;
		float d = offset;
		if (it->parent && level < it->parent->ll->level) // Negative line
			d = -d;
		it->ll->level += d;
		assert(0 <= it->ll->level && it->ll->level <= 255);
	}
}

void output_BW_result(const unsigned char* inIm, float offset, const string& binarized_output, const int w, const int h);
bool output_shape(const vector<vector<Segment> >& segments_pieces, int& num_C, int& num_L, int w, int h, Rect R, const std::string& fileName);
bool output_svg_path(const vector<vector<Segment> >& segments_pieces, int& num_C, int& num_L, int w, int h, Rect R, const std::string& fileName);

#endif