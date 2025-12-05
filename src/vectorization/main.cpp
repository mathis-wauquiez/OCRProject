/**
 * @file main.cpp
 * @brief convert png images to a binary svg
 * @author Yuchen He <yuchenroy@sjtu.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "lltree.h"
#include "gass.h"
#include "curv.h"
#include "image.h"
#include "xmtime.h"
#include "io_png.h"
#include <algorithm>
#include <fstream>
#include <cmath>
#include <string>
#include <stdlib.h>
#include <time.h>
#include "utility.h"
#include "affine_sp_vectorization.h"

using namespace std;

/* ------ MAIN ------- */

/// Main procedure for curvature microscope.
int main(int argc, char** argv) {
	float offset, scale, global_thresh;
	int refine_iteration;
	string BW_output_path, result1, result2, result3, result4;
	size_t w, h;
	vector<vector<Segment> > outline_pieces;
	vector<vector<Point>> pxl_boundary;
	vector<bool> positive;
	int num_C = 0, num_L = 0;
	int seg_num = 0;

	readCmdLine(argc, argv, offset, scale, global_thresh, refine_iteration,
		BW_output_path, result1, result2,
		result3, result4); // parse the input arguments
	unsigned char* original_img = readImage(argv, &w, &h);
	const Rect R = { MARGIN, MARGIN, (int)w, (int)h }; // specify the margin
	float initial_scale = scale + K * ds;
	// K*ds is the largest scale of affine scale space, from which we back trace along the flow
	output_BW_result(original_img, offset, BW_output_path, w, h);

	cout << "------ Start to Vectorize -----" << endl;
	clock_t c_start = clock();

	extract_pxl_boundary(pxl_boundary, scale, original_img, w, h, offset); // [Algorithm 4 Part I]
	split_boundary_by_corners(pxl_boundary, initial_scale, outline_pieces); // [Algorithm 4 Part II]
	vectorize_boundary(outline_pieces); // [Algorithm 4 Part III line 20 - 22]
	vector<vector<Segment> > refined_pieces;

	refine_splitting(outline_pieces, refined_pieces, global_thresh, false); // [Algorithm 4 Part III line 23 - 42]
	
	vectorize_boundary(refined_pieces);
	for (int i = 0; i < refined_pieces.size(); i++)
	{
		seg_num += (int)refined_pieces[i].size();
	}
	cout << "Number of Bezier segments (without merging): " << seg_num << endl;
	clock_t c_end = clock();
	float time1 = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "Vectorization takes time: " << time1 << " ms." << endl;

	output_svg_path(refined_pieces, num_C, num_L, w, h, R, result1);
	output_shape(refined_pieces, num_C, num_L, w, h, R, result2);

	cout << "------ Start to Merge -----" << endl;
	c_start = clock();
	refine_splitting(outline_pieces, refined_pieces, global_thresh, true); // Simplification described in Section 2.3.1
	vector<vector<Segment> > iterated_pieces;
	for (int iter = 0; iter < refine_iteration - 1; iter++)
	{
		iterated_pieces = refined_pieces;
		refine_splitting(iterated_pieces, refined_pieces, global_thresh, true);
	}
	vectorize_boundary(refined_pieces);
	c_end = clock();
	seg_num = 0;
	for (int i = 0; i < refined_pieces.size(); i++)
	{
		seg_num += (int)refined_pieces[i].size();
	}
	cout << "Number of Bezier segments (with merging): " << seg_num << endl;
	c_end = clock();
	time1 = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "Merging takes time: " << time1 << " ms." << endl;
	output_svg_path(refined_pieces, num_C, num_L, w, h, R, result3);
	output_shape(refined_pieces, num_C, num_L, w, h, R, result4);

	free(original_img);
	return 0;
}