cmake_minimum_required(VERSION 3.14)
project(image_similarity)
find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 17)

add_executable(
        image_similarity
        ImageAnalyzer.h
        ImageAnalyzer.cpp
        KeyPoint.h
        KeyPoint.cpp
        main.cpp
        )
target_link_libraries( image_similarity ${OpenCV_LIBS} )