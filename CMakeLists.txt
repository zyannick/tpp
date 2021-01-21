cmake_minimum_required(VERSION 3.5)

project(tpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tpp main.cpp)

find_package(PkgConfig)

INCLUDE_DIRECTORIES ( "/usr/local/include/" )
INCLUDE_DIRECTORIES ( "/usr/local/include/opencv4/" )
INCLUDE_DIRECTORIES ( "/usr/include/eigen3/" )
INCLUDE_DIRECTORIES ( . )

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if( NOT FFTW_ROOT AND DEFINED ENV{FFTWDIR} )
    set( FFTW_ROOT $ENV{FFTWDIR} )
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)

#Determine from PKG
if( PKG_CONFIG_FOUND AND NOT FFTW_ROOT )
    pkg_check_modules( PKG_FFTW QUIET "fftw3" )
endif()


LINK_DIRECTORIES("/usr/lib")
TARGET_LINK_LIBRARIES(tpp fftw3f fftw3 stdc++fs)


find_package(OpenCV REQUIRED )
INCLUDE_DIRECTORIES ( "${OpenCV_INCLUDE_DIRS}" )
target_link_libraries( tpp ${OpenCV_LIBS} )
#target_link_libraries(${PROJECT_NAME} stdc++fs)


set_property(TARGET tpp PROPERTY CXX_STANDARD 20)
set_property(TARGET tpp PROPERTY CXX_STANDARD_REQUIRED ON)


set(CMAKE_CXX_STANDARD 20)
enable_testing()
