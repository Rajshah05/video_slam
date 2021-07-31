cmake_minimum_required(VERSION 3.0)

project(video_slam)
add_compile_options(-std=c++17)

find_package(OpenCV REQUIRED)
find_package(SDL2 REQUIRED)
find_package(Eigen3 REQUIRED)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/dep/g2o/cmake_modules)
# set (G2O_ROOT  /user/local)
find_package(G2O REQUIRED)


add_executable(video_slam src/main.cpp src/frame.cpp)
target_link_libraries(video_slam ${SDL2_LIBRARIES})
target_link_libraries(video_slam ${OpenCV_LIBS})
target_link_libraries(video_slam Eigen3::Eigen)
target_include_directories(video_slam PUBLIC include)