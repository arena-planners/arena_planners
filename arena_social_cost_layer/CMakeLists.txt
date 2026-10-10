cmake_minimum_required(VERSION 3.8)
project(arena_social_cost_layer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(arena_people_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_library(arena_social_cost_layer SHARED
  src/social_cost_layer.cpp
)
target_include_directories(arena_social_cost_layer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(arena_social_cost_layer
  rclcpp
  nav2_costmap_2d
  pluginlib
  arena_people_msgs
  geometry_msgs
)

pluginlib_export_plugin_description_file(nav2_costmap_2d arena_social_cost_layer.xml)

install(TARGETS arena_social_cost_layer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/
  DESTINATION include
)
install(FILES arena_social_cost_layer.xml
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_cost_math test/test_cost_math.cpp)
  target_include_directories(test_cost_math PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
endif()

ament_export_include_directories(include)
ament_export_libraries(arena_social_cost_layer)
ament_export_dependencies(rclcpp nav2_costmap_2d pluginlib arena_people_msgs geometry_msgs)

ament_package()
