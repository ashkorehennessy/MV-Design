cmake_minimum_required(VERSION 3.20)
project(yolo_fastestv2)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fopenmp")

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/inc)

# 查找 CPU 架构
execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCHITECTURE
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 根据架构设置可执行文件名称
if (ARCHITECTURE STREQUAL "aarch64")
    set(EXECUTABLE_NAME yolo_fastestv2_aarch64)
elseif (ARCHITECTURE STREQUAL "x86_64")
    set(EXECUTABLE_NAME yolo_fastestv2_x86_64)
    link_directories(${CMAKE_SOURCE_DIR}/lib)
else ()
    message(FATAL_ERROR "Unsupported architecture: ${ARCHITECTURE}")
endif ()

# 源文件
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# 构建可执行文件
add_executable(${EXECUTABLE_NAME} ${SOURCES})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ncnn ${OpenCV_LIBS})
