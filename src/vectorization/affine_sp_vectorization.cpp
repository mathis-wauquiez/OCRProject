/**
 * @file affine_sp_vectorization.cpp
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

#include "affine_sp_vectorization.h"

void extract_pxl_boundary(vector<vector<Point>>& pxl_boundary, const float scale,
	unsigned char* original_img, const int w, const int h, const float offset)
{
	LLTree tree(original_img, w, h, offset, 255.0f, 1);
	fix_level(tree, offset);

	if (scale > 0) {
		const int size = (int)tree.nodes().size();
		for (int i = 0; i < size; i++)
		{
			vector<Point>& poly_line = tree.nodes()[i].ll->line;
			smooth(poly_line, scale);
			pxl_boundary.push_back(poly_line);
		}
	}
}

void split_boundary_by_corners(const vector<vector<Point>>& pxl_boundary,
	const float initial_scale, vector<vector<Segment> >& outline_pieces
)
{
	for (int i = 0; i < pxl_boundary.size(); i++)
	{
		vector<Point> initial_outline = pxl_boundary[i];
		if (initial_outline.size() < 10)
			continue;

		vector<vector<Point> > candidate_corner_sequence;
		vector<vector<Point> > candidate_normal_sequence;
		std::vector<Point> vertices;
		std::vector<Point> normal_dir;
		vector<int> cutting_index;
		vector<Point> single_outline = pxl_boundary[i];
		smooth(single_outline, initial_scale);
		corner_detection(single_outline, vertices, normal_dir, cutting_index);
		vector<bool> active_index;

		for (int j = 0; j < vertices.size(); j++) {
			active_index.push_back(true);
			vector<Point> initial_corner;
			vector<Point> initial_normal;
			initial_corner.push_back(vertices[j]);
			initial_normal.push_back(normal_dir[j]);
			candidate_corner_sequence.push_back(initial_corner);
			candidate_normal_sequence.push_back(initial_normal);
		}
		float newscale = initial_scale;

		for (int t = 0; t < K; t++) {
			newscale -= ds;
			vector<Point> single_outline = pxl_boundary[i];
			smooth(single_outline, newscale);
			(void)inv_corner_trace(single_outline, candidate_corner_sequence, candidate_normal_sequence, active_index);
		}
		vector<int> corner_index = inv_corner_trace(initial_outline, candidate_corner_sequence, candidate_normal_sequence, active_index);
		int num_vertices = corner_index.size();
		if (num_vertices == 0) // no vertex identified, check if circle by 4piA \approx P^2?
		{
			if (check_if_circle(initial_outline)) // is indeed a circle
			{
				vector<Segment> input;
				input.push_back(Segment(initial_outline, true));
				outline_pieces.push_back(input);
				continue;
			}
			vector<int> most_distant_indices = find_indices_degenerate(initial_outline);
			corner_index.push_back(most_distant_indices[0]);
			corner_index.push_back(most_distant_indices[1]);
		}
		split_by_corner_index(corner_index, initial_outline, outline_pieces);
	}
}

void split_by_corner_index(vector<int>& corner_index, vector<Point>& initial_outline, vector<vector<Segment> >& outline_pieces)
{
	vector<Segment> inv_outline;
	int num_vertices = corner_index.size();
	sort(corner_index.begin(), corner_index.end());
	corner_index.erase(unique(corner_index.begin(), corner_index.end()), corner_index.end());

	num_vertices = corner_index.size();

	for (int j = 0; j < num_vertices - 1; j++)
	{
		vector<Point>::iterator it_begin = initial_outline.begin() + corner_index[j];
		vector<Point>::iterator it_end = initial_outline.begin() + corner_index[j + 1] + 1;
		vector<Point> segment_nodes;
		for (auto it = it_begin; it != it_end; it++)
		{
			segment_nodes.push_back(*it);
		}
		inv_outline.push_back(Segment(segment_nodes));
	}
	vector<Point> segment_nodes;
	vector<Point>::iterator it_begin = initial_outline.begin() + corner_index[num_vertices - 1];
	vector<Point>::iterator it_end = initial_outline.end();
	for (auto it = it_begin; it != it_end; it++)
	{
		segment_nodes.push_back(*it);
	}
	it_begin = initial_outline.begin();
	it_end = initial_outline.begin() + corner_index[0] + 1;
	for (auto it = it_begin; it != it_end; it++)
	{
		segment_nodes.push_back(*it);
	}

	inv_outline.push_back(Segment(segment_nodes));
	outline_pieces.push_back(inv_outline);
}

void vectorize_boundary(vector<vector<Segment> >& outline_pieces)
{
	for (int i = 0; i < outline_pieces.size(); i++)
	{
		for (int j = 0; j < outline_pieces[i].size(); j++)
		{
			Segment& segment = outline_pieces[i][j];
			if (segment.if_circle())
			{
				segment.identify_circle();
			}
			else if (segment.size() > 4)
			{
				segment.cubic_Bezier_fitting();
			}
			else
			{
				segment.set_straight();
			}
		}
	}
}