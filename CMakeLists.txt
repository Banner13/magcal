cmake_minimum_required(VERSION 3.10)

# 项目名称和语言
project(Magcal LANGUAGES CXX)

# 检查依赖库 Eigen 是否存在
find_package(Eigen3 REQUIRED)

# 源文件
set(SOURCES magcal.cpp)

# 生成动态库
add_library(MagcalShared SHARED ${SOURCES})
target_link_libraries(MagcalShared Eigen3::Eigen)

# 生成静态库
add_library(MagcalStatic STATIC ${SOURCES})
target_link_libraries(MagcalStatic Eigen3::Eigen)

# 指定库输出目录
set_target_properties(MagcalShared MagcalStatic PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 如果需要，可以定义安装目标
install(TARGETS MagcalShared MagcalStatic
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
