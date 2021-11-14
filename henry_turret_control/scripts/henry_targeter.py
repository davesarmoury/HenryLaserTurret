from niryo_robot_arm_commander.msg import RobotMoveActionGoal
from geometry_msgs.msg import PoseStamped
import rospy
import math

from control_msgs.msg import FollowJointTrajectoryActionGoal

def mover():
    pub = rospy.Publisher('/niryo_robot_follow_joint_trajectory_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
    rospy.init_node('mover', anonymous=True)
    rate = rospy.Rate(2)

    for i in range(50):
        msg = FollowJointTrajectoryActionGoal()
        msg.goal.cmd.joints = [math.sin(rospy.get_time() / 10.0),0,0,0,0,0]

        if rospy.is_shutdown():
            break
        rospy.loginfo(str(msg.goal.cmd.joints))
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        mover()
    except rospy.ROSInterruptException:
        pass

#################################################################
#niryo@niryo_pi4:~$ rostopic pub /niryo_robot_arm_commander/robot_action/goal niryo_robot_arm_commander/RobotMoveActionGoal "header:
#  seq: 0
#  stamp:
#    secs: 0
#    nsecs: 0
#  frame_id: ''
#goal_id:
#  stamp:
#    secs: 0
#    nsecs: 0
#  id: ''
#goal:
#  cmd:
#    cmd_type: 0
#    joints: [0,0,0.2,0,0,0]
#    position: {x: 0.0, y: 0.0, z: 0.0}
#    rpy: {roll: 0.0, pitch: 0.0, yaw: 0.0}
#    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}
#    shift: {axis_number: 0, value: 0.0}
#    list_poses:
#    - position: {x: 0.0, y: 0.0, z: 0.0}
#      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}
#    dist_smoothing: 0.0" 
##################################################################
#/where_in_the_world_is_henry_san_diego
#davesarmoury@armoury-agx:~/ws/henry_ws/src/HenryLaserTurret/henry_turret_control/scripts$ rosmsg show geometry_msgs/PoseStamped 
#std_msgs/Header header
#  uint32 seq
#  time stamp
#  string frame_id
#geometry_msgs/Pose pose
#  geometry_msgs/Point position
#    float64 x
#    float64 y
#    float64 z
#  geometry_msgs/Quaternion orientation
#    float64 x
#    float64 y
#    float64 z
#    float64 w
#################################################################
