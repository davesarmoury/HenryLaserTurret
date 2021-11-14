cmake_minimum_required(VERSION 3.0.2)
project(henry_turret_control)

find_package(catkin REQUIRED COMPONENTS)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES henry_turret_control
#  CATKIN_DEPENDS geometry_msgs niryo_robot_arm_commander trac_ik_kinematics_plugin
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
