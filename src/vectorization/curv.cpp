/**
 * @file curv.cpp
 * @brief functions related to curve operations
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

#include "curv.h"
#include "levelLine.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cassert>

 /// Sign (+/-1) of \a x
#define SIGN(x) (((x)>0.0)?1: -1)

/// Distance between points.
static float dist(Point p, const Point& q) {
	p.x -= q.x; p.y -= q.y;
	return sqrt(p.x * p.x + p.y * p.y);
}

static float dist2(Point p, const Point& q) {
	p.x -= q.x; p.y -= q.y;
	return p.x * p.x + p.y * p.y;
}

inline float det(const Point& a, const Point& b, const Point& c) {
	return -((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

inline int modulo(int i, int n) {
	if (i < 0)  return i + n;
	if (i >= n) return i - n;
	return i;
}

void curvature_filter(vector<float>& curvature_on_curve)
{
	int n = curvature_on_curve.size();
	for (int i = 0; i < 20; i++)
	{
		for (int k = 0; k < n; k++)
		{
			int center = k;
			int pre, post, pre1, post1;

			pre = modulo(k - 1, n); post = modulo(k + 1, n);
			pre1 = modulo(k - 2, n); post1 = modulo(k + 2, n);

			float cur_k = curvature_on_curve[center];
			float cur_k_minus_1 = curvature_on_curve[pre];
			float cur_k_plus_1 = curvature_on_curve[post];
			float cur_k_minus_2 = curvature_on_curve[pre1];
			float cur_k_plus_2 = curvature_on_curve[post1];
			float val = (8 * cur_k + 4 * cur_k_minus_1 + 4 * cur_k_plus_1 + 1 * cur_k_minus_2 + 1 * cur_k_plus_2) / 18;
			curvature_on_curve[center] = val;
		}
	}
}

void compute_curvature(const std::vector<Point>& curve, vector<float>& curvature_on_curve)
{
	int n = curve.size();
	const Point* q = &curve[n - 1];
	const Point* p = &curve[0];
	const Point* r = &curve[1];
	float u = dist(*p, *q);

	for (int k = 0; k < n; k++, r++) {
		if (k + 1 == n)
			r = &curve[0];
		float v = dist(*p, *r);
		float d = u * v * dist(*q, *r);
		float K = ((d == 0) ? 0 : -2 * det(*p, *q, *r) / d); // curvature

		curvature_on_curve.push_back(K);
		q = p;
		p = r;
		u = v;
	}
}

void corner_detection(const std::vector<Point>& curve, std::vector<Point>& vertices, std::vector<Point>& normal_dir, vector<int>& index)
{
	// Compute the curvature
	if (curve.size() < 3)
		return;

	int n = curve.size();

	std::vector <float> curvature_on_curve;

	compute_curvature(curve, curvature_on_curve);

	curvature_filter(curvature_on_curve);

	// EXTREMAL FILTERING
	for (int k = 0; k < n; k++)
	{
		int center = k;
		int pre, post, pre1, post1;
		if (k + 1 == n)
		{
			pre = k - 1;
			post = 0;
		}
		else if (k == 0)
		{
			pre = n - 1;
			post = 1;
		}
		else
		{
			pre = k - 1;
			post = k + 1;
		}

		if (k + 2 == n)
		{
			pre1 = k - 2;
			post1 = 0;
		}
		else if (k + 1 == n)
		{
			pre1 = k - 1;
			post1 = 1;
		}
		else if (k - 1 == 0)
		{
			pre1 = n - 1;
			post1 = 3;
		}
		else if (k == 0)
		{
			pre1 = n - 2;
			post1 = 2;
		}
		else
		{
			pre1 = k - 2;
			post1 = k + 2;
		}

		float cur_k = curvature_on_curve[center];
		float cur_k_minus_1 = curvature_on_curve[pre];
		float cur_k_plus_1 = curvature_on_curve[post];
		float cur_k_minus_2 = curvature_on_curve[pre1];
		float cur_k_plus_2 = curvature_on_curve[post1];
		float abs_cur_k = fabs(cur_k);
		if (abs_cur_k > fabs(cur_k_minus_1) and abs_cur_k > fabs(cur_k_plus_1)
			and abs_cur_k > fabs(cur_k_minus_2) and abs_cur_k > fabs(cur_k_plus_2) and
			abs_cur_k > 0.001)
		{
			float x, y;
			x = curve[post].x - curve[pre].x;
			y = curve[post].y - curve[pre].y;

			float norm = sqrt(x * x + y * y);
			float Nx = y / norm;
			float Ny = (-x / norm);
			int sign = cur_k >= 0 ? +1 : -1;
			normal_dir.push_back(Point(-sign * Nx, -sign * Ny));
			vertices.push_back(Point(curve[k].x, curve[k].y));
			index.push_back(k);
		}
	}
}

vector<int> inv_corner_trace(const std::vector<Point>& curve, vector<vector<Point> >& current_corner_sequence, vector<vector<Point> >& current_normal_sequence,
	std::vector<bool>& active_index)
{
	std::vector<Point> candidate_vertices;
	std::vector<Point> candidate_normals;
	std::vector<int> candidate_index;
	vector<int> corner_index;

	corner_detection(curve, candidate_vertices, candidate_normals, candidate_index);

	int cand_corner_num = candidate_vertices.size();

	for (int i = 0; i < active_index.size(); i++) {
		if (!active_index[i])
			continue;

		Point active_corner = current_corner_sequence[i].back();
		Point active_normal = current_normal_sequence[i].back();
		float corner_x = active_corner.x;
		float corner_y = active_corner.y;
		float normal_x = active_normal.x;
		float normal_y = active_normal.y;

		int found_index = -1;

		float alpha = 9.0 / 10.0;
		float D = sqrt(10.0);

		for (int j = 0; j < cand_corner_num; j++) {
			float x = candidate_vertices[j].x;
			float y = candidate_vertices[j].y;
			float distance_to_corner = sqrt((x - corner_x) * (x - corner_x) + (y - corner_y) * (y - corner_y));
			if (!(distance_to_corner > D)) {
				float cos_angle_to_normal = ((x - corner_x) * normal_x + (y - corner_y) * normal_y) / distance_to_corner;
				if (cos_angle_to_normal < alpha) {
					continue;
				}
				else {
					alpha = cos_angle_to_normal;
					found_index = j;
				}
			}
		}

		if (found_index >= 0) {
			current_corner_sequence[i].push_back(candidate_vertices[found_index]);
			current_normal_sequence[i].push_back(candidate_normals[found_index]);
			corner_index.push_back(candidate_index[found_index]);
		}
		else {
			active_index[i] = false;
		}
	}
	return corner_index;
}

void Segment::identify_circle()
{
	assert(is_closed_);

	Point P0 = vertices_[0];
	Point P1 = vertices_[ceil(vertices_.size() / 3.0)];
	assert(vertices_.size() >= 3);
	Point P2 = vertices_[ceil(2.0 * vertices_.size() / 3.0)];

	float x12 = P0.x - P1.x;
	float x13 = P0.x - P2.x;

	float y12 = P0.y - P1.y;
	float y13 = P0.y - P2.y;

	float y31 = P2.y - P0.y;
	float y21 = P1.y - P0.y;

	float x31 = P2.x - P0.x;
	float x21 = P1.x - P0.x;

	float sx13 = P0.x * P0.x - P2.x * P2.x;
	float sx21 = P1.x * P1.x - P0.x * P0.x;
	float sy13 = P0.y * P0.y - P2.y * P2.y;
	float sy21 = P1.y * P1.y - P0.y * P0.y;

	float f = ((sx13) * (x12)
		+(sy13) * (x12)
		+(sx21) * (x13)
		+(sy21) * (x13))
		/ (2 * ((y31) * (x12)-(y21) * (x13)));

	float g = ((sx13) * (y12)
		+(sy13) * (y12)
		+(sx21) * (y13)
		+(sy21) * (y13))
		/ (2 * ((x31) * (y12)-(x21) * (y13)));

	float c = -P0.x * P0.x - P0.y * P0.y - 2 * g * P0.x - 2 * f * P0.y;

	control_points_.clear();
	control_points_.push_back(Point(-g, -f));
	radius_ = sqrt(g * g + f * f - c);
}

void Segment::compute_par()
{
	par_.clear();
	par_.push_back(0.0);
	float accum_len = 0.0;
	for (int i = 1; i < vertices_.size(); i++)
	{
		accum_len += dist(vertices_[i], vertices_[i - 1]);
		par_.push_back(accum_len);
	}
	for (int i = 0; i < vertices_.size(); i++)
	{
		par_[i] = par_[i] / accum_len;
	}
}

void Segment::cubic_Bezier_fitting()
{
	this->compute_par();
	this->update_control_points();
}

float Segment::cubic_Bezier_fitting(int& splitting_index)
{
	// // Initialization
	this->compute_par();
	this->update_control_points();

	if (vertices_.size() == 2)
		return 0.0;

	// measure error
	vector<Point> ctr_points = this->get_control_points();
	Point P0 = ctr_points[0];
	Point P1 = ctr_points[2];
	Point P2 = ctr_points[3];
	Point P3 = ctr_points[1];

	vector<Point> curve_points;
	for (int i = 0; i < vertices_.size(); i++)
	{
		float ti = par_[i];
		curve_points.push_back(powf(1 - ti, 3) * P0 + 3 * ti * powf(1 - ti, 2) * P1 + 3 * ti * ti * (1 - ti) * P2 + powf(ti, 3) * P3);
	}

	float max_error = 0;

	for (int i = 1; i < vertices_.size()-1; i++)
	{
		float max_error_i = FLT_MAX;
		Point test_point = vertices_[i];
		for (int j = 1; j < vertices_.size()-1; j++)
		{
			Point curve_point = curve_points[j];
			float error = dist2(curve_point, test_point);
			if (error < max_error_i)
			{
				max_error_i = error;
			}
		}

		if (max_error_i > max_error)
		{
			max_error = max_error_i;
			splitting_index = i;
		}
	}

	return sqrt(max_error);
}

float Segment::max_abs_curvature()
{
	float max_abs_curv = 0.0;
	Point A = control_points_[0];
	Point B = control_points_[2];
	Point C = control_points_[3];
	Point D = control_points_[1];

	Point A1(3 * (B.x - A.x), 3 * (B.y - A.y));
	Point B1(3 * (C.x - B.x), 3 * (C.y - B.y));
	Point C1(3 * (D.x - C.x), 3 * (D.y - C.y));

	Point A2(2 * (B1.x - A1.x), 2 * (B1.y - A1.y));
	Point B2(2 * (C1.x - B1.x), 2 * (C1.y - B1.y));
	for (int i = 0; i < vertices_.size(); i++)
	{
		float t = par_[i];
		float Bp_x = (A1.x - 2 * B1.x + C1.x) * t * t + (2 * B1.x - 2 * A1.x) * t + A1.x;
		float Bp_y = (A1.y - 2 * B1.y + C1.y) * t * t + (2 * B1.y - 2 * A1.y) * t + A1.y;
		float Bpp_x = (B2.x - A2.x) * t + A2.x;
		float Bpp_y = (B2.y - A2.y) * t + A2.y;

		float cur_abs_curv = fabs((Bp_x * Bpp_y - Bp_y * Bpp_x) / powf(Bp_x * Bp_x + Bp_y * Bp_y, 1.5));

		if (cur_abs_curv > max_abs_curv)
		{
			max_abs_curv = cur_abs_curv;
		}
	}

	return max_abs_curv;
}

void Segment::update_control_points()
{
	float a1(0), a2(0), a12(0);
	Point C1(0.0, 0.0), C2(0.0, 0.0);
	Point P0 = control_points_[0];
	Point P3 = control_points_[1];

	if (vertices_.size() == 2)
	{
		control_points_.clear();
		control_points_.push_back(P0);
		control_points_.push_back(P3);
		control_points_.push_back(0.5*(P0+P3));
		control_points_.push_back(0.5*(P0+P3));
		return;
	}

	if (vertices_.size() == 3)
	{
		control_points_.clear();
		control_points_.push_back(P0);
		control_points_.push_back(P3);
		control_points_.push_back(vertices_[1]);
		control_points_.push_back(vertices_[1]);
		return;
	}
	
	for (int i = 0; i < vertices_.size(); i++)
	{
		Point pi = vertices_[i];
		float si = par_[i], si2 = si * si, si3 = si * si2, si4 = si2 * si2;
		float ti = 1 - si, ti2 = ti * ti, ti3 = ti * ti2, ti4 = ti2 * ti2;

		a1 += si2 * ti4;
		a2 += si4 * ti2;
		a12 += si3 * ti3;
		C1 += (3 * si * ti2) * (pi - (ti3 * P0) - (si3 * P3));
		C2 += (3 * si2 * ti) * (pi - (ti3 * P0) - (si3 * P3));
	}

	a1 *= 9;
	a2 *= 9;
	a12 *= 9;
	double denom = a1 * a2 - a12 * a12;
	Point P1((a2 * C1.x - a12 * C2.x) / denom, (a2 * C1.y - a12 * C2.y) / denom); /// A1 --> A12
	Point P2((a1 * C2.x - a12 * C1.x) / denom, (a1 * C2.y - a12 * C1.y) / denom);

	control_points_.clear();
	control_points_.push_back(P0);
	control_points_.push_back(P3);
	control_points_.push_back(P1);
	control_points_.push_back(P2);
}

inline bool is_not_flat(Point p1, Point p2) {
	return (fabs(p1.x * p2.x + p1.y * p2.y + 1) >= 0.1);
}

inline float norm2(Point p) {
	return sqrt(p.x * p.x + p.y * p.y);
}

vector<int> find_indices_degenerate(const vector<Point>& curve_points)
{
	vector<Point> convex_outline;
	convex_outline = convex_hull(curve_points);
	vector<Point> most_distant_points = rotate_calipers(convex_outline);
	return search_index(curve_points, most_distant_points);
}

vector<Point> tangents_from_Bezier(vector<Segment>& outline)
{
	vector<Point> tangent_sequence;
	int num_segments = outline.size();

	for (int j = 0; j < num_segments; j++) // generate the tangent sequence
	{
		outline[j].cubic_Bezier_fitting();
		vector<Point> ctrl_points = outline[j].get_control_points();

		Point tangent = ctrl_points[2] - ctrl_points[0];
		tangent_sequence.push_back(tangent / (norm2(tangent) + 1e-10));
		tangent = ctrl_points[3] - ctrl_points[1];
		tangent_sequence.push_back(tangent / (norm2(tangent) + 1e-10));
	}
	return tangent_sequence;
}

void split_by_tangent_angles(vector<Segment>& new_segments, const int start_index, const vector<Segment>& outline, const vector<Point>& tangent_sequence)
{
	int num_segments = outline.size();
	int corner_counter = 0;
	vector<Point> new_segment_points; // container

	for (int j = start_index; j < start_index + num_segments + 1; j++)
	{
		int index = j < num_segments ? j : (j - num_segments);
		int post = 2 * index;
		int pre = index == 0 ? (2 * num_segments - 1) : (2 * index - 1);
		if (is_not_flat(tangent_sequence[pre], tangent_sequence[post]))
			corner_counter += 1;

		if (corner_counter == 1) // first encounter a new segment
		{
			vector<Point> segment_points = outline[index].get_points();
			for (int k = 0; k < segment_points.size() - 1; k++)
			{
				new_segment_points.push_back(segment_points[k]);
			}
		}
		else // split by the corner
		{
			new_segment_points.push_back(outline[index].get_control_points()[0]);
			new_segments.push_back(Segment(new_segment_points));
			new_segment_points.clear();
			corner_counter = 0;
			j = j - 1;
		}
	}
}

void split_by_distant_points(vector<Segment>& new_segments, const vector<Point>& circle_points)
{
	vector<Point> new_segment_points;
	vector<int> most_distant_indices = find_indices_degenerate(circle_points);
	sort(most_distant_indices.begin(), most_distant_indices.end());
	for (int k = most_distant_indices[1]; k < circle_points.size(); k++)
		new_segment_points.push_back(circle_points[k]);
	for (int k = 0; k < most_distant_indices[0] + 1; k++)
		new_segment_points.push_back(circle_points[k]);
	new_segments.push_back(Segment(new_segment_points));
	new_segment_points.clear();
	for (int k = most_distant_indices[0]; k < most_distant_indices[1] + 1; k++)
		new_segment_points.push_back(circle_points[k]);
	new_segments.push_back(Segment(new_segment_points));
}

void split_to_reduce_error(vector<Segment>& updated_segments, const float error_thresh)
{
	float global_max_error = error_thresh + 10.0;
	int count = 0;
	vector<Segment> new_segments;

	while (global_max_error > error_thresh)
	{
		global_max_error = 0;

		for (int j = 0; j < updated_segments.size(); j++)
		{
			Segment segment = updated_segments[j];

			int max_err_ind;
			float error = segment.cubic_Bezier_fitting(max_err_ind);

			if (count == 0)
			{
				count += 1;
				global_max_error = error;
			}
			else
			{
				if (error > global_max_error)
				{
					global_max_error = error;
				}
			}

			if (error < error_thresh)
			{
				new_segments.push_back(segment);
				continue;
			}

			const vector<Point>& full_points = segment.get_points();
			vector<Point> partial_points(full_points.begin(),
				full_points.begin() + max_err_ind + 1);

			Segment partial_segment(partial_points);

			new_segments.push_back(partial_segment);

			partial_points.clear();

			for (int k = max_err_ind; k < full_points.size(); k++)
			{
				partial_points.push_back(full_points[k]);
			}
			partial_segment = Segment(partial_points);
			new_segments.push_back(partial_segment);
		}
		updated_segments = new_segments;

		new_segments.clear();
	}
}

void simplify_segments(vector<vector<Segment> >& refined_pieces, vector<Segment>& updated_segments, const float error_thresh)
{
	vector<Point> all_pts;
	vector<int> new_splitting_indices;

	int cum_sum = 0;
	int cur_pos, pre_pos, next_pos;
	int newseg_num = updated_segments.size();

	if (newseg_num <= 3)
	{
		refined_pieces.push_back(updated_segments);
		return;
	}

	for (int j = 0; j < newseg_num; j++)
	{
		new_splitting_indices.push_back(cum_sum);
		const vector<Point>& seg_pts = updated_segments[j].get_points();
		int segment_size = seg_pts.size();
		cum_sum += segment_size;
		all_pts.insert(all_pts.end(), seg_pts.begin(), seg_pts.end());
	}

	int total_length = all_pts.size();
	int final_index = new_splitting_indices.size() - 1;
	int cur_index = 0;
	int pre_index = final_index;
	int next_index = 1;
	int original_final = final_index;

	for (int j = 0; j <= original_final; j++)
	{
		cur_pos = new_splitting_indices[cur_index];
		pre_pos = new_splitting_indices[pre_index];
		next_pos = new_splitting_indices[next_index];

		vector<Point> cand_pts_to_merge;
		if (pre_pos > cur_pos or next_pos < cur_pos)
		{
			for (int k = pre_pos; k < total_length; k++)
				cand_pts_to_merge.push_back(all_pts[k]);
			for (int k = 0; k <= next_pos; k++)
				cand_pts_to_merge.push_back(all_pts[k]);
		}
		else
		{
			for (int k = pre_pos; k <= next_pos; k++)
				cand_pts_to_merge.push_back(all_pts[k]);
		}

		Segment cand_merge_seg(cand_pts_to_merge);
		int max_err_ind;
		float error = cand_merge_seg.cubic_Bezier_fitting(max_err_ind);

		if (error >= error_thresh)
		{
			pre_index = cur_index;
			cur_index = next_index;
			next_index = (next_index + 1 > final_index ? 0 : next_index + 1);
		}
		else
		{
			final_index -= 1;
			new_splitting_indices.erase(new_splitting_indices.begin() + cur_index);
			if (cur_index == final_index + 1)
			{
				pre_index = final_index;
				cur_index = 0;
				next_index = 1;
			}
			if (new_splitting_indices.size() <= 2)
				break;
		}
	}
	vector<Segment> simplified_seg;
	for (int j = 0; j < new_splitting_indices.size(); j++)
	{
		vector<Point> new_seg_pts;

		int ini_index = new_splitting_indices[j];
		int end_index;

		if (j == new_splitting_indices.size() - 1)
		{
			end_index = new_splitting_indices[0];
		}
		else
		{
			end_index = new_splitting_indices[j + 1];
		}

		if (ini_index > end_index)
		{
			for (int k = ini_index; k < total_length; k++)
				new_seg_pts.push_back(all_pts[k]);
			for (int k = 0; k < end_index; k++)
				new_seg_pts.push_back(all_pts[k]);
		}
		else
		{
			for (int k = ini_index; k < end_index + 1; k++)
				new_seg_pts.push_back(all_pts[k]);
		}

		simplified_seg.push_back(Segment(new_seg_pts));
	}

	refined_pieces.push_back(simplified_seg);
}

void refine_splitting(const vector<vector<Segment> >& original_pieces, vector<vector<Segment> >& refined_pieces, const float error_thresh, const bool simplified)
{
	refined_pieces.clear();
	vector<int> circle_index;

	for (int i = 0; i < original_pieces.size(); i++)
	{
		vector<Segment> outline = original_pieces[i];
		int num_segments = outline.size();

		if (outline[0].if_circle())
		{
			circle_index.push_back(i);
			continue;
		}

		vector<Point> tangent_sequence = tangents_from_Bezier(outline);

		int start_index = -1;

		for (int j = 0; j < num_segments; j++)
		{
			int post = 2 * j;
			int pre = j == 0 ? (2 * num_segments - 1) : (2 * j - 1);
			if (is_not_flat(tangent_sequence[pre], tangent_sequence[post]))
			{
				start_index = j;
				break;
			}
		}

		vector<Segment> updated_segments;
		
		if (start_index != -1)
		{
			split_by_tangent_angles(updated_segments, start_index, outline, tangent_sequence);
		}
		else
		{
			vector<Segment> circle_segments = original_pieces[i];
			vector<Point> circle_points;
			for (int k = 0; k < circle_segments.size(); k++)
			{
				vector<Point> points = circle_segments[k].get_points();
				for (int l = 0; l < points.size(); l++)
					circle_points.push_back(points[l]);
			}
			if (check_if_circle(circle_points))
			{
				circle_segments.clear();
				circle_segments.push_back(Segment(circle_points, true));
				refined_pieces.push_back(circle_segments);
				continue;
			}
			split_by_distant_points(updated_segments, circle_points);
		}
		
		split_to_reduce_error(updated_segments, error_thresh);

		if (simplified)
		{
			simplify_segments(refined_pieces, updated_segments, error_thresh);
		}
		else
		{
			refined_pieces.push_back(updated_segments);
		}
	}

	for (int i = 0; i < circle_index.size(); i++)
	{
		refined_pieces.push_back(original_pieces[circle_index[i]]);
	}
}

float cross(const Point& O, const Point& A, const Point& B)
{
	return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

float area(const Point& O, const Point& A, const Point& B)
{
	return fabs(cross(O, A, B)) / 2.0;
}

vector<Point> convex_hull(vector<Point> P)
{
	size_t n = P.size(), k = 0;
	if (n <= 3) return P;
	vector<Point> H(2 * n);

	// Sort points lexicographically
	sort(P.begin(), P.end());

	// Build lower hull
	for (size_t i = 0; i < n; ++i) {
		while (k >= 2 && cross(H[k - 2], H[k - 1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}

	// Build upper hull
	for (size_t i = n - 1, t = k + 1; i > 0; --i) {
		while (k >= t && cross(H[k - 2], H[k - 1], P[i - 1]) <= 0) k--;
		H[k++] = P[i - 1];
	}

	H.resize(k - 1);
	return H;
}

vector<Point> rotate_calipers(const vector<Point>& P)
{
	vector<Point> most_distant_pairs;
	float max_distance2 = 0.0;
	int max_i, max_j;
	int i0 = P.size() - 1;
	int N = P.size();
	int i = 0;
	int j = i + 1;
	int j0;
	while (area(P[i], P[(i + 1 == N) ? 0 : (i + 1)], P[(j + 1 == N) ? 0 : (j + 1)]) > area(P[i], P[(i + 1 == N) ? 0 : (i + 1)], P[j]))
	{
		j = (j + 1 == N) ? 0 : (j + 1);
		j0 = j;
	}
	if (i0 == j0)
	{
		most_distant_pairs.push_back(P[0]);
		most_distant_pairs.push_back(P[i0]);
		return most_distant_pairs;
	}
	while (j != i0 and i != j0)
	{
		i = (i + 1 == N) ? 0 : (i + 1);

		float distance2 = (P[j].x - P[i].x) * (P[j].x - P[i].x) + (P[j].y - P[i].y) * (P[j].y - P[i].y);
		if (distance2 > max_distance2)
		{
			max_distance2 = distance2;
			max_i = i;
			max_j = j;
		}
		while (area(P[i], P[(i + 1 == N) ? 0 : (i + 1)], P[(j + 1 == N) ? 0 : (j + 1)]) > area(P[i], P[(i + 1 == N) ? 0 : (i + 1)], P[j]))
		{
			j = (j + 1 == N) ? 0 : (j + 1);
			if (i != j0 or j != i0)
			{
				float distance2 = (P[j].x - P[i].x) * (P[j].x - P[i].x) + (P[j].y - P[i].y) * (P[j].y - P[i].y);
				if (distance2 > max_distance2)
				{
					max_distance2 = distance2;
					max_i = i;
					max_j = j;
				}
			}
			else
			{
				break;
			}
		}

		if (area(P[j], P[(i + 1 == N) ? 0 : (i + 1)], P[(j + 1 == N) ? 0 : (j + 1)]) == area(P[i], P[(i + 1 == N) ? 0 : (i + 1)], P[j]))
		{
			if (i != j0 or j != i0)
			{
				float distance2 = (P[(j + 1 == N) ? 0 : (j + 1)].x - P[i].x) * (P[(j + 1 == N) ? 0 : (j + 1)].x - P[i].x) + (P[(j + 1 == N) ? 0 : (j + 1)].y - P[i].y) * (P[(j + 1 == N) ? 0 : (j + 1)].y - P[i].y);
				if (distance2 > max_distance2)
				{
					max_distance2 = distance2;
					max_i = i;
					max_j = (j + 1 == N) ? 0 : (j + 1);
				}
			}
			else
			{
				float distance2 = (P[j].x - P[(i + 1 == N) ? 0 : (i + 1)].x) * (P[j].x - P[(i + 1 == N) ? 0 : (i + 1)].x) + (P[j].y - P[(i + 1 == N) ? 0 : (i + 1)].y) * (P[j].y - P[(i + 1 == N) ? 0 : (i + 1)].y);
				if (distance2 > max_distance2)
				{
					max_distance2 = distance2;
					max_i = (i + 1 == N) ? 0 : (i + 1);
					max_j = j;
				}
			}
		}
	}

	most_distant_pairs.push_back(P[max_i]);
	most_distant_pairs.push_back(P[max_j]);
	return most_distant_pairs;
}

vector<int> search_index(const vector<Point>& original_curve, const vector<Point>& point_coordinates)
{
	vector<int> indices;
	for (int i = 0; i < point_coordinates.size(); i++)
	{
		int min_index;
		Point point = point_coordinates[i];
		float min_dist2 = std::numeric_limits<float>::max();
		for (int j = 0; j < original_curve.size(); j++)
		{
			Point curve_point = original_curve[j];
			float distance2 = (curve_point.x - point.x) * (curve_point.x - point.x) + (curve_point.y - point.y) * (curve_point.y - point.y);
			if (distance2 < min_dist2)
			{
				min_dist2 = distance2;
				min_index = j;
			}
		}
		indices.push_back(min_index);
	}
	return indices;
}

bool check_if_circle(const vector<Point>& initial_outline)
{
	float area = 0.0;
	Point P0 = initial_outline[0];
	for (int j = 1; j < initial_outline.size() - 1; j++)
	{
		Point P1 = initial_outline[j];
		Point P2 = initial_outline[j + 1];
		area += fabs((P1.x - P0.x) * (P2.y - P0.y) - (P2.x - P0.x) * (P1.y - P0.y)) / 2.0;
	}

	// compute the perimeter
	float perimeter = 0.0;
	for (int j = 0; j < initial_outline.size() - 1; j++)
	{
		Point P1 = initial_outline[j];
		Point P2 = initial_outline[j + 1];
		perimeter += sqrt((P2.x - P1.x) * (P2.x - P1.x) + (P2.y - P1.y) * (P2.y - P1.y));
	}

	return fabs(area * 4 * M_PI / (perimeter * perimeter) - 1) < 0.005;
}