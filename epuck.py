#!/usr/bin/env python
import rospy
import numpy as np
import math
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


from collections import deque

CALIBRATION_CONSTANT = 0.041148

def handle_camera_message(msg):
    for epuck in EPUCKS:
        epuck.update_odometry(msg)
        

class EpuckROS:
    def __init__(self, name, vision):
        self.name = name
        self.vision = vision
        self.history = deque(maxlen=999)
        self.twist_message = Twist
        self.odometry_subscriber = rospy.Subscriber(
            self.name + "/odom", Odometry, self.save_state)
        self.odometry_update_publisher = rospy.Publisher(
            self.name + "/odom_reset", Odometry)
        self.twist_publisher = rospy.Publisher(
            self.name + "/base_link/cmd_vel", Odometry)
        self.current_state = Odometry

    def save_state(self, msg):
        self.history.append(msg)
        self.current_state = msg

    def send_twist(self, v, omega):
        self.twist_message.linear.x = v
        self.twist_message.linear.y = 0
        self.twist_message.linear.z = 0
        self.twist_message.angular.x = 0
        self.twist_message.angular.y = 0
        self.twist_message.angular.z = omega
        if self.twist_publisher != None:
            self.twist_publisher.publish(self.twist_message)

    # to be the callback of the Kalman Filter that takes in everything, for now we just update the internal state of the bot
    def update_odometry(self, img):
        if not self.vision.calibrated:
            return self.vision.find_epuck()
        (x, y, theta) = self.vision.find_epuck()
        msg = Odometry
        msg.header.frame_id = "base_link"
        msg.header.stamp = rospy.Time().now()
        msg.pose.pose.x = x
        msg.pose.pose.y = y
        msg.pose.pose.z = 0
        q = tf.transformations.quaternion_from_euler(0, 0, theta)
        msg.pose.pose.orientation = Quaternion(*q)
        msg.twist.twist.linear.x = 0
        msg.twist.twist.omega.z = 0
        self.odometry_update_publisher.publish(msg)


class EpuckVision:
    def __init__(self, name, lower_hsv=(0, 0, 0), upper_hsv=(255, 255, 255)):
        self.name = name
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.history = deque(maxlen=999)
        self.calibrated = False
        self.calibration_factor = deque(maxlen=10)
        self.frames_out = 0
        self.bridge = CvBridge()

    def find_epuck(self, img):
        cvimg = self.bridge.imgmsg_to_cv(img, "bgr8")
        cvimg = cv2.GaussianBlur(cvimg, (5, 5), 0)
        hsv = cv2.cvtColor(cvimg, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.dilate(mask, None, iterations=1)
        _, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)
        cnts = [(i, x) for i, x in enumerate(cnts) if cv2.contourArea(x) > 10]
        if len(cnts) > 0:
            circles = (seq(cnts)
                       # Index , Contour , Circle
                       .map(lambda x: (x[0], x[1], cv2.minEnclosingCircle(x[1])))
                       # Index, Contour, Radius
                       .map(lambda x: (x[0], x[1], x[2][1]))
                       .filter(lambda x:  0.5 * x[2] * cv2.arcLength(x[1], True) / cv2.contourArea(x[1]) < 1.5)
                       .map(lambda x: (x[0], x[1])))

            if circles.len() == 2:
                self.frames_out = 0
                # if it has internal contour we put it first TODO: update mask and put bigger circle, sort by size
                cir = circles.sorted(key=lambda x: hierarchy[0][x[0]][2])
                cir = [x[1] for x in cir]
                M1 = cv2.moments(cir[0])
                M2 = cv2.moments(cir[1])
                # check hierarchy
                center1 = np.array(
                    [int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])], dtype="int32")
                center2 = np.array(
                    [int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])], dtype="int32")
                vision_center = (center1 + center2) / 2
                distance = np.linalg.norm(center1 - center2)
                self.calibration_factor.append(CALIBRATION_CONSTANT / distance)
                if len(self.calibration_factor) == 10:
                    self.calibrated = True
                    factor = np.sum(self.calibration_factor) / 10
                    true_center = vision_center * factor
                    self.history.append((rospy.Time.now(), true_center))
                    return true_center

        self.frames_out += 1
        if self.frames_out > 4:
            self.calibrated = False
            self.calibration_factor = deque(maxlen=10)
        elif len(self.history) > 0:
            return self.history[-1][1]

        return None

global EPUCKS
rospy.init("ros_handler")
epuck_vision_red = EpuckVision("epuck_0", (0, 10, 100), (7, 255, 255))
epuck_vision_blue = EpuckVision("epuck_1", (90, 79, 100), (96, 255, 255))
rospuck = EpuckROS("epuck_0", epuck_vision_red)
EPUCKS = [rospuck]
rospy.Subscriber(
            "cv_camera_node/image_raw", Image, handle_camera_message)


