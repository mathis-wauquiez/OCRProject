Binary Shape Vectorization by Affine Scale-space
by Yuchen He <yuchenroy@sjtu.edu.cn>

This software converts any raster image to a binary image by thresholding, then renders the vectorized result.

[1] IPOL demo, https://ipolcore.ipol.im/demo/clientApp/demo.html?id=401

Build
-----
Prerequisites: CMake version 2.6 or later, libPNG

-  Windows with MinGW:
$ cd /path_to_this_file/
$ mkdir build && cd build && cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
$ cmake --build.

-  Linux and MacOS:
$ cd /path_to_this_file/
$ mkdir build . && cd build && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
$ cmake --build .

- Test:
$ main.exe ../data/butterfly.png

Usage
-----
main.exe in.png [options]
  - in.png: PNG input image
Options:
  -f <float>: threshold for binarization (default 127.5)
  -s <float>: scale of smoothing by affine scale-space (default 2.0)
  -T <float>: threshold for the approximating accuracy in pixel (default 1.0)
  -R <int>: number of iteration for the refinement (default 0.0)
  -B <fileName.png>: output file for the binarized image which is to be vectorized
  -v <fileName>: outline without merging
  -V <fileName>: shape without merging
  -o <fileName>: outline after merging
  -O <fileName>: shape after merging

Files to be reviewed
-----
CMakeLists.txt
curv.cpp curv.h
utility.cpp utility.h
affine_sp_vectorization.cpp affine_sp_vectorization.h
main.cpp

Some codes are adapted from the codes from [2]

[2] Ciomaga, A., Monasse, P., & Morel, J. M. (2017). The image curvature microscope: Accurate curvature computation at subpixel resolution. Image Processing On Line, 7, 197-217.
