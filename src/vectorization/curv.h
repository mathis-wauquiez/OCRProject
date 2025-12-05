/**
 * @file curv.h
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
#ifndef CURVE_H
#define CURVE_H

#include "levelLine.h"
#include <cfloat>
#include <vector>

using namespace std;

class Segment
{
public:

	Segment(vector<Point> vertices, bool is_closed = false) : vertices_(vertices), is_closed_(is_closed) {
		control_points_.push_back(vertices.front());
		control_points_.push_back(vertices.back());
	}
	// constructor
	vector<Point>::iterator P0() { return vertices_.begin(); }
	int size() const { return vertices_.size(); }
	void identify_circle();
	/**
	 * \brief Identify the center and radius of a circle given 4 points.
	 * This is used in the degenerate case when the closed curve is identified as
	 * a circle. See Section 2.2
	 */
	void compute_par();
	/**
	 * \brief compute the arc-length parametrization of a curve. [Algorithm 3 line 2]
	 */
	float cubic_Bezier_fitting(int& splitting_index);
	/**
	 * \brief Approximate the segment by a single Bezier curve, output the error, and find the point
	 * on the curve to be approximated furthest away from the Bezier curve.
	 *
	 * \param splitting_index [output]: the index of curve point furthest away from the approximating Bezier curve
	 */
	void cubic_Bezier_fitting();
	/**
	 * \brief Compute the control points of approximating Bezier curve. [Algorithm 3 line 2]
	 */
	float max_abs_curvature();
	/**
	 * \brief Compute the absolute curvatures and output the maximum. [Algorithm 1]
	 */
	const vector<Point>& get_control_points() const { return control_points_; }
	const vector<Point>& get_points() const { return vertices_; }
	const vector<float>& get_par() const { return par_; }
	bool if_circle() const { return is_closed_; }
	float get_radius() const { return radius_; }
	void set_straight() { is_straight_ = true; }
	bool if_straight() const { return is_straight_; }
private:
	vector<Point> vertices_;
	vector<float> par_;
	vector<Point> control_points_;
	void update_control_points();
	/**
	 * \brief Actual implementation of the computation process for Bezier fitting [Algorithm 3].
	 */
	bool is_closed_ = false;
	bool is_straight_ = false;
	float radius_; // only effective if is_closed = true;
};

void curvature_filter(vector<float>& curvature_on_curve);
/**
 * \brief Filter the curvatures by a moving average. [Algorithm 1 line 1]
 *
 * \param curvature_on_curve [input/output]: a vector of curvature values to be smoothed in place
 */
void compute_curvature(const std::vector<Point>& curve, vector<float>& curvature_on_curve);
/**
 * \brief Compute discrete curvatures [Algorithm 4 line 6]
 *
 * \param curve: curves whose curvatures are to be computed
 * \param curvature_on_curve [output]: the output curvature values
 */
void corner_detection(const vector<Point>& curve, vector<Point>& vertices, vector<Point>& normal_dir, vector<int>& index);
/**
 * \brief Detect the corners by identifying the curvature extrema. [Algorithm 1]
 *
 * \param curve: curves whose corners and normal directions to be identified
 * \param vertices [output]:  coordinates of identified curvature extrema
 * \param normal_dir [output]: normal direction on curve points
 * \param index [output]: indices of identified curvature extrema
 */
vector<int> inv_corner_trace(const vector<Point>& curve, vector<vector<Point> >& current_corner_sequence, vector<vector<Point> >& current_normal_sequence,
	vector<bool>& active_index);
/**
 * \brief Trace the de-pixelized corners of a curve reversely in the affine scale space. [Algorithm 4 Part II]
 *
 * \param curve: the curve whose corners to be identified
 * \param current_corner_sequence [output]: sequence of points of corners being updated as tracing back the flow
 * \param current_normal_sequence [output]: sequence of normal directions at corners being updated as tracing back the flow
 * \param active_index [output]: a record of indices of points remain corners as tracing back the flow
 */
void refine_splitting(const vector<vector<Segment> >& original_pieces, vector<vector<Segment> >& refined_pieces, const float error_thresh, const bool simplified);
/**
 * \brief Refine the segments to satisfy the error threshold requirenment [Algorithm 4 Part III]
 *
 * \param original_pieces: pieces of segments to be fitted by Bezier curves
 * \param refined_pieces [output]: refined segments
 * \param error_thresh: error threshold to be satisfied by the approximation
 * \param simplified: simplify by merging or not
 */
float cross(const Point& O, const Point& A, const Point& B);
/**
 * \brief compute the norm of the cross product of (A-O) and (B-O).
 */
vector<Point> tangents_from_Bezier(vector<Segment>& outline);
/**
 * \brief Compute the tangents of Bezier curves.
 *
 * \param outline: collection of Bezier curves whose tangents to be computed
 */
void split_by_tangent_angles(vector<Segment>& new_segments, const int start_index, const vector<Segment>& outline, const vector<Point>& tangent_sequence);
/**
 * \brief Split into two segments if the one-sided tangents form non-flat angles [Algorithm 4 line 24-26]
 *
 * \param new_segments [output]: the output new segments
 * \param start_index: the index of the first segment in the closed curve
 * \param outline: the collection of segments to be splitted
 * \param tangent_sequence: sequence of tangents
 */
void split_by_distant_points(vector<Segment>& new_segments, const vector<Point>& circle_points);
/**
 * \brief Split a degenerate curve by its most distant points. [Algorithm 2]
 *
 * \param new_segments [output]: two segments of the input circle points split by the most distant points
 * \param circle_points: closed curve to be splitted
 */
void split_to_reduce_error(vector<Segment>& updated_segments, const float error_thresh);
/**
 * \brief Split a curve into two segments if the approximation error by a single Bezier curve is above error_thresh [Algorithm 4 line 34]
 */
void simplify_segments(vector<vector<Segment> >& refined_pieces, vector<Segment>& updated_segments, const float error_thresh);
/**
 * \brief Simplify the splitting by merging segments with flat tangents while keeping error below the error_thresh. [Section 2.3.1 Algorithm 4 line 25]
 *
 * \param refined_pieces [input]: pieces to be simplified
 * \param updated_segments [output]: simplified pieces
 * \param error_thresh: error threshold to be satisfied
 */
vector<int> find_indices_degenerate(const vector<Point>& curve_points);
/**
 * \brief Output the indices of most distant points.
 */
vector<Point> convex_hull(vector<Point> P);
/**
 * \brief Get the convex hull of a collection of points P
 */
vector<Point> rotate_calipers(const vector<Point>& P);
/**
 * \brief Implementation of the rotate calipers to find the most distant points on a closed curve [Algorithm 2]
 */
vector<int>	search_index(const vector<Point>& original_curve, const vector<Point>& point_coordinates);
/**
 * \brief Given point's coorinate, find the index on a curve.
 */
bool check_if_circle(const vector<Point>& initial_outline);
/**
 * \brief Check if the degenerate curve is indeed a circle. [Algorithm 2 line 2]
 */

#endif
