#!/usr/bin/env python

import rospy
import math
import tf
import numpy
from tf.transformations import *
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionGoal
from trac_ik_python.trac_ik import IK

laser_height_shift = 0.2
reach_dist = 0.4
horizontal_offset = 0.3
horizontal_period = 4.0

ik_solver = None
pub = None
seed_state = [0.0, 0.0, 0.0, 0.0, -1.5707, 0.0]

limits_lower = [-1.5, -1.5707, -1.3, -2.0,  -2.0, -3.14]
limits_upper = [ 1.5,     0.6,  1.3,  2.0,   0.0,  3.14]

def pos_callback(msg):
    global ik_solver, pub, seed_state, broadcaster, laser_o_to_cam_mtx, world_to_laser_o

    broadcaster.sendTransform((msg.pose.position.x,msg.pose.position.y,msg.pose.position.z),
                              (0,0,0,1),
                              rospy.Time.now(),
                              "Fuzzman",
                              "zed2_left_camera_frame")

    h_mtx = compose_matrix(translate=(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z))
    o_to_h_mtx = numpy.matmul(laser_o_to_cam_mtx, h_mtx)

    t = rospy.get_time()
    offset_mtx = compose_matrix(translate=(math.sin(t / horizontal_period) * horizontal_offset, math.cos(t / horizontal_period) * horizontal_offset, 0.0))
    o_to_h_mtx = numpy.matmul(o_to_h_mtx, offset_mtx)

    scale, shear, angles, translate, perspective = decompose_matrix(o_to_h_mtx)

    horizontal_distance = math.sqrt(translate[0]*translate[0] + translate[1]*translate[1])
    yaw = math.atan2(translate[1], translate[0])
    pitch = math.atan2(translate[2],horizontal_distance) * -1.0 + 1.5707
    angles_shifted = (0, pitch, yaw)

    h_dist = math.sqrt(translate[0] * translate[0] + translate[1] * translate[1] + translate[2] * translate[2])
    translate_shifted = (translate[0] / h_dist * reach_dist, translate[1] / h_dist * reach_dist, translate[2] / h_dist * reach_dist)

    l_to_h_mtx = compose_matrix(translate=translate_shifted, angles=angles_shifted)
    w_to_l_mtx = numpy.matmul(world_to_laser_o, l_to_h_mtx)

    scale, shear, angles, pos, perspective = decompose_matrix(w_to_l_mtx)
    ori = quaternion_from_euler(angles[0], angles[1], angles[2])

#    broadcaster.sendTransform(translate,
#                              quaternion_from_euler(angles[0], angles[1], angles[2]),
#                              rospy.Time.now(),
#                              "laser",
#                              "world")

    ik = ik_solver.get_ik(seed_state, pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], ori[3])

    if ik:
        msg = FollowJointTrajectoryActionGoal()
        msg.header.stamp = rospy.Time.now()
        msg.goal_id.stamp = rospy.Time.now()
        msg.goal_id.id = "henry_" + str(rospy.get_time())

        msg.goal.trajectory.header.frame_id = "world"
        msg.goal.trajectory.joint_names = list(ik_solver.joint_names)

        point_msg = JointTrajectoryPoint()
        point_msg.positions = list(ik)
        point_msg.time_from_start = rospy.Duration.from_sec(1.0)

        msg.goal.trajectory.points.append(point_msg)

        pub.publish(msg)

    else:
        rospy.logwarn("IK_ERROR")

def mover():
    global broadcaster, laser_o_to_cam_mtx, world_to_laser_o, ik_solver, pub
    rospy.init_node('mover', anonymous=True)

    rospy.loginfo("Getting IK")
    ik_solver = IK("world", "laser_link", timeout=0.05, solve_type="Distance")
    ik_solver.set_joint_limits(limits_lower, limits_upper)

    rospy.loginfo("Initializing TF Broadcaster and Listener")
    broadcaster = tf.TransformBroadcaster()
    listener = tf.TransformListener()

    rospy.loginfo("Recording Static Offsets")
    listener.waitForTransform('world', 'zed2_left_camera_frame', rospy.Time(0), rospy.Duration(0.5))
    trans, rot = listener.lookupTransform('world', 'zed2_left_camera_frame', rospy.Time(0))
    trans[2] = trans[2] - laser_height_shift
    laser_o_to_cam_mtx = compose_matrix(translate=trans, angles=euler_from_quaternion(rot))

    world_to_laser_o = compose_matrix(translate=(0,0,laser_height_shift), angles=(0,0,0))

    pub = rospy.Publisher('/niryo_robot_follow_joint_trajectory_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
    rospy.Subscriber('/where_in_the_world_is_henry_san_diego', PoseStamped, pos_callback)

    rospy.loginfo("Let the games begin!")
    rospy.spin()

if __name__ == '__main__':
    try:
        mover()
    except rospy.ROSInterruptException:
        pass
