cmake_minimum_required(VERSION 3.8)
project(hearo_speech_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/TTSGenerate.srv"
  "srv/GetOrCreateAudioId.srv"
  DEPENDENCIES std_msgs builtin_interfaces
)

install(DIRECTORY hearo_speech_manager/
  DESTINATION lib/python3.12/site-packages/hearo_speech_manager
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN "srv" EXCLUDE
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
