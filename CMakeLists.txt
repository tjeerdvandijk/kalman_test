cmake_minimum_required(VERSION 3.10)
project(kalman_imgui)

set(CMAKE_CXX_STANDARD 17)

file(GLOB IMGUI_SRC
    external/imgui/*.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

file(GLOB IMPLOT_SRC
    external/implot/*.cpp
)

add_executable(kalman_imgui
    kalman.cpp
    ${IMGUI_SRC}
    ${IMPLOT_SRC}
)

target_include_directories(kalman_imgui PRIVATE
    external/imgui
    external/imgui/backends
    external/implot
)

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)

pkg_check_modules(GLFW REQUIRED glfw3)

target_link_libraries(kalman_imgui PRIVATE ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} GL GLEW glfw Eigen3::Eigen)
