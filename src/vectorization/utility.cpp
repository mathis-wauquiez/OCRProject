/**
 * @file utility.cpp
 * @brief Utility functions
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

#include "utility.h"
#include <cstring>
static const int BLACK = 0;
static const int WHITE = 255;

void readCmdLine(int& argc, char** argv, float& offset, float& scale, float& global_thresh, int& refine_iteration,
	string& BW_output_path, string& result1, string& result2,
	string& result3, string& result4)
{
	if (argc < 2)
	{
		cout << "No input image file is specified." << endl;
		exit(0);
	}
	scale = 2.0f;
	global_thresh = 1.0f;
	refine_iteration = 0;
	offset = 127.5f;
	CmdLine cmd;
	cmd.add(make_option('f', offset));
	cmd.add(make_option('s', scale));
	cmd.add(make_option('T', global_thresh));
	cmd.add(make_option('R', refine_iteration));
	cmd.add(make_option('v', result1));
	cmd.add(make_option('V', result2));
	cmd.add(make_option('o', result3));
	cmd.add(make_option('O', result4));
	cmd.add(make_option('B', BW_output_path));
	cmd.process(argc, argv);
}

unsigned char* readImage(char** argv, size_t* pW, size_t* pH)
{
	unsigned char* readIm = io_png_read_u8_gray(argv[1], pW, pH);
	if (!readIm) {
		std::cerr << "Unable to open image " << argv[1] << std::endl;
		exit(1);
	}

	int total_size = (*pW) * (*pH);
	unsigned char* inImage = new unsigned char[total_size];
	memcpy(inImage, readIm, total_size * sizeof(unsigned char));
	free(readIm);
	return inImage;
}

void output_BW_result(const unsigned char* inIm, float offset, const string& binarized_output, const int w, const int h)
{
	unsigned char* binary = new unsigned char[w * h];
	for (int i = 0; i < w * h; i++)
	{
		binary[i] = (inIm[i] > offset) ? WHITE : BLACK;
	}

	io_png_write_u8(binarized_output.c_str(), binary, w, h, 1);
	delete[] binary;
}

bool output_shape(const vector<vector<Segment> >& segments_pieces, int& num_C, int& num_L, int w, int h, Rect R, const std::string& fileName)
{
	std::ofstream file(fileName.c_str());

	file << "<svg xmlns=\"http://www.w3.org/2000/svg\" ";
	file << "width=\"" << w << "\" " << "height=\"" << h << "\" ";
	file << "viewBox=\"0 0 " << w << ' ' << h << "\">" << std::endl;
	file << "<path stroke=\"none\" fill=\"black\" fill-rule=\"evenodd\" d=\"";

	vector<int> circle_index;
	for (int i = 0; i < segments_pieces.size(); i++)
	{
		vector<Segment> segments = segments_pieces[i];
		if (segments.begin()->if_circle())
		{
			circle_index.push_back(i);
			continue;
		}

		file << "M ";

		for (auto it = segments.begin(); it != segments.end(); ++it)
		{
			vector<Point> controlP = it->get_control_points();
			Point P = controlP[0];
			float max_abs_curv;
			if (not it->if_straight())
			{
				max_abs_curv = it->max_abs_curvature();
			}
			else
			{
				max_abs_curv = 0.0;
			}

			if (max_abs_curv > 0.001)
			{
				num_C += 1;
				file << P.x - R.x << ' ' << P.y - R.y << " C ";
				P = controlP[2];
				file << P.x - R.x << ' ' << P.y - R.y << ", ";
				P = controlP[3];
				file << P.x - R.x << ' ' << P.y - R.y << ", ";
				if (it == segments.end() - 1)
				{
					P = controlP[1];
					file << P.x - R.x << ' ' << P.y - R.y << "Z";
				}
			}
			else
			{
				num_L += 1;
				file << P.x - R.x << ' ' << P.y - R.y << "L";
				P = controlP[1];
				file << P.x - R.x << ' ' << P.y - R.y << ' ';
			}
		}
	}

	for (int i = 0; i < circle_index.size(); i++)
	{
		Segment circle = segments_pieces[circle_index[i]][0];
		Point center = circle.get_control_points()[0];
		float radius = circle.get_radius();
		file << "M " << center.x - R.x - radius << ' ' << center.y - R.y << " a ";
		file << radius << " , " << radius << " 0 1,1 " << radius * 2 << ",0 ";
		file << radius << " , " << radius << " 0 1,1 " << -radius * 2 << ",0 Z ";
	}
	file << "\" />";

	file << "</svg>" << std::endl;
	return file.good();
}

bool output_svg_path(const vector<vector<Segment> >& segments_pieces, int& num_C, int& num_L, int w, int h, Rect R, const std::string& fileName) {
	std::ofstream file(fileName.c_str());
	file << "<svg xmlns=\"http://www.w3.org/2000/svg\" ";
	file << "width=\"" << w << "\" " << "height=\"" << h << "\" ";
	file << "viewBox=\"0 0 " << w << ' ' << h << "\">" << std::endl;
	file << "<path stroke=\"black\" stroke-width=\"0.5\" fill=\"none\"  d=\"";

	vector<int> circle_index;
	for (int i = 0; i < segments_pieces.size(); i++)
	{
		vector<Segment> segments = segments_pieces[i];

		if (segments.begin()->if_circle())
		{
			circle_index.push_back(i);
			continue;
		}

		file << "M ";

		for (auto it = segments.begin(); it != segments.end(); ++it)
		{
			vector<Point> controlP = it->get_control_points();
			Point P = controlP[0];

			float max_abs_curv;
			if (not it->if_straight())
			{
				max_abs_curv = it->max_abs_curvature();
			}
			else
			{
				max_abs_curv = 0.0;
			}

			if (max_abs_curv > 0.001)
			{
				num_C += 1;
				file << P.x - R.x << ' ' << P.y - R.y << " C ";
				P = controlP[2];
				file << P.x - R.x << ' ' << P.y - R.y << ", ";
				P = controlP[3];
				file << P.x - R.x << ' ' << P.y - R.y << ", ";
				if (it == segments.end() - 1)
				{
					P = controlP[1];
					file << P.x - R.x << ' ' << P.y - R.y << "Z";
				}
			}
			else
			{
				num_L += 1;
				file << P.x - R.x << ' ' << P.y - R.y << "L";
				P = controlP[1];
				file << P.x - R.x << ' ' << P.y - R.y << ' ';
			}
		}
	}

	file << "\" />";

	for (int i = 0; i < circle_index.size(); i++)
	{
		Segment circle = segments_pieces[circle_index[i]][0];
		Point center = circle.get_control_points()[0];
		float radius = circle.get_radius();
		file << "<circle cx=\"" << center.x - R.x << "\" cy=\"" << center.y - R.y << "\" r=\"" << radius << "\" stroke=\"black\" stroke-width=\"0.5\" fill=\"none\" />";
	}

	for (int i = 0; i < segments_pieces.size(); i++)
	{
		vector<Segment> segments = segments_pieces[i];
		for (auto it = segments.begin(); it != segments.end(); ++it)
		{
			if (it->if_circle())
			{
				continue;
			}
			vector<Point> ctr_points = it->get_control_points();
			Point center = ctr_points[0];
			file << "<circle cx=\"" << center.x - R.x << "\" cy=\"" << center.y - R.y << "\" r=\"4\" fill=\"red\" />";
		}
	}

	for (int i = 0; i < circle_index.size(); i++)
	{
		Segment circle = segments_pieces[circle_index[i]][0];
		Point center = circle.get_control_points()[0];

		file << "<circle cx=\"" << center.x - R.x << "\" cy=\"" << center.y - R.y << "\" r=\"4\" fill=\"blue\" />";
	}

	file << "</svg>" << std::endl;
	return file.good();
}