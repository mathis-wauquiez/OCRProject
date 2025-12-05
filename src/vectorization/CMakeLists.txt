cmake_minimum_required(VERSION 2.6)
set(CMAKE_CXX_STANDARD 14)

find_package(ZLIB)
include_directories(${ZLIB_INCLUDE_DIRS})
find_package(PNG)
include_directories(${PNG_INCLUDE_DIRS})
add_definitions(${PNG_DEFINITIONS})

set(SRC
    cmdLine.h
    curv.cpp curv.h
    gass.cpp gass.h
    image.cpp image.h
    io_png.c io_png.h
    levelLine.cpp levelLine.h
    lltree.cpp lltree.h
    utility.cpp utility.h
    affine_sp_vectorization.cpp affine_sp_vectorization.h
    main.cpp
    xmtime.h)
add_executable(main ${SRC})
target_link_libraries(main ${PNG_LIBRARIES})

find_package(OpenMP)
if(OPENMP_FOUND)
    set_target_properties(main PROPERTIES
                          COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
endif(OPENMP_FOUND)

if(UNIX)
    if(OPENMP_FOUND)
        set(CMAKE_EXE_LINKER_FLAGS  ${OpenMP_CXX_FLAGS})
    endif(OPENMP_FOUND)    
endif(UNIX)
