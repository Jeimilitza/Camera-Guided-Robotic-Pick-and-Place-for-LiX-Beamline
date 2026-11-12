cmake_minimum_required(VERSION 3.8)
project(arm_gripper_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)

install(
   DIRECTORY urdf launch config
   DESTINATION share/${PROJECT_NAME}
)

install(
   PROGRAMS
   src/pickNplace.py
   src/open_gripper.py
   src/place_screwdriver.py
   src/place_hole.py
   DESTINATION lib/${PROJECT_NAME}
)

ament_package()
