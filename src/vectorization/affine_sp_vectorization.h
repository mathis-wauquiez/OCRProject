/**
 * @file affine_sp_vectorization.h
 * @brief Main functions for affine-scale-space based vectorization
 * @author Yuchen He <yuchenroy@sjtu.edu.cn>
 *
 * All rights reserved.
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

#ifndef AFFINE_SP_VECTORIZATION_H
#define AFFINE_SP_VECTORIZATION_H

#include "lltree.h"
#include "utility.h"

static const double ds = 0.5; // Delta sigma, smoothing parameter for each step in the sequence of affine flow
static const int K = 4; // number of scale spaces during the affine shortening

void extract_pxl_boundary(vector<vector<Point>>& pxl_boundary, const float scale,
	unsigned char* original_img, const int w, const int h, const float offset);
/**
 * \brief Extract pixel level boundary as polygonal lines. [Algorithm 4 Part I]
 *
 * \param pxl_boundary [output]: collection of polygonal lines extracted from the shapes
 * \param scale: smoothness parameter applied to the boundaries
 * \param original_img: orginal raster image
 * \param w: image width
 * \param h: image height
 * \param offset: threshold for image binarization
 */
void split_boundary_by_corners(const vector<vector<Point>>& pxl_boundary,
	const float initial_scale, vector<vector<Segment> >& outline_pieces);
/**
 * \brief Split the boundaries by the identified corners. [Algorithm 4 Part II]
 *
 * \param pxl_boundary: collection of polygonal closed curves.
 * \param initial_scale: the smoothness applied to the boundary curve, which is then splitted
 * \param outline_pieces [output]: collection of curve segments whose end points are identified corners
 */
void split_by_corner_index(vector<int>& corner_index, vector<Point>& initial_outline,
	vector<vector<Segment> >& outline_pieces);
/**
 * \brief Split the boundaries by the given indices.
 *
 * \param corner_index: indices of the initial_outline to be split
 * \param initial_outline: the curve to be splitted
 * \param outline_pieces [output]: the splitted curves
 */
void vectorize_boundary(vector<vector<Segment> >& outline_pieces);
/**
 * \brief Vectorize the boundary curves. [Algorithm 4 Part III]
 *
 * \param: outline_pieces [output]: each segment will be vectorized via Bezier approximation or being replaced by a circle
 * Each segment will store its own control points after running this.
 */
#endif