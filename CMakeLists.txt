cmake_minimum_required(VERSION 3.0)

project(video_slam)

find_package(OpenCV REQUIRED)
find_package(SDL2 REQUIRED)

add_executable(video_slam src/main.cpp src/featureExtractorAndMatcher.cpp)
target_link_libraries(video_slam ${SDL2_LIBRARIES})
target_link_libraries(video_slam ${OpenCV_LIBS})
target_include_directories(video_slam PUBLIC include)