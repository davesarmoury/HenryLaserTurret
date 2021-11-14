#!/usr/bin/env python

import rospy
import math
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionGoal

def mover():
    pub = rospy.Publisher('/niryo_robot_follow_joint_trajectory_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
    rospy.init_node('mover', anonymous=True)
    rate = rospy.Rate(2)

    for i in range(100):
        msg = FollowJointTrajectoryActionGoal()
        msg.header.stamp = rospy.Time.now()
        msg.goal_id.stamp = rospy.Time.now()
        msg.goal_id.id = "henry_" + str(rospy.get_time())

        msg.goal.trajectory.header.frame_id = "world"
        msg.goal.trajectory.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

        point_msg = JointTrajectoryPoint()
        point_msg.positions = [math.sin(rospy.get_time() /6.0),0,0,0,0,0]
        point_msg.time_from_start = rospy.Duration.from_sec(1.0)

        msg.goal.trajectory.points.append(point_msg)

        if rospy.is_shutdown():
            break
        rospy.loginfo(str(point_msg.positions))
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        mover()
    except rospy.ROSInterruptException:
        pass
