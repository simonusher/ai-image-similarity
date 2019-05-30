cmake_minimum_required(VERSION 3.14)
project(image_similarity)
find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS program_options)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
add_executable(
        image_similarity
        ImageAnalyzer.h
        ImageAnalyzer.cpp
        KeyPoint.h
        KeyPoint.cpp
        main.cpp
        )
target_link_libraries( image_similarity ${OpenCV_LIBS} )
target_link_libraries(image_similarity Boost::program_options)