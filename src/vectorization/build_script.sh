#!/bin/bash

# Binary Shape Vectorization by Affine Scale-space
# Build script for Linux/MacOS

echo "=========================================="
echo "Binary Shape Vectorization - Build Script"
echo "=========================================="
echo ""

# Remove existing build directory
if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
    echo "✓ Build directory removed"
else
    echo "No existing build directory found"
fi

echo ""

# Create new build directory
echo "Creating build directory..."
mkdir build
cd build

echo "✓ Build directory created"
echo ""

# Run CMake
echo "Running CMake configuration..."
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed!"
    exit 1
fi

echo "✓ CMake configuration successful"
echo ""

# Build the project
echo "Building the project..."
cmake --build .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Build completed successfully!"
echo "=========================================="
echo ""
echo "To test the application, run:"
echo "  ./build/main ../data/butterfly.png"
echo ""
