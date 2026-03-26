ф#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

from custom_interfaces.msg import IMUmsg
from custom_interfaces.srv import MotorSRV, ServSRV

from cv_bridge import CvBridge
from standard_devices.weblog import LoggerMixin


class UserNode(Node, LoggerMixin):

    def __init__(self):
        super().__init__("user_node")
        # обязательно поменяйте, если тестируете не на 3 рботе
        self.robot_id = "robot3"

        self._init_weblogger()
        self._log("START")

        # камера
        self.bridge = CvBridge()
        self.pub = self.create_publisher(
            Image, f"/{self.robot_id}/user_camera", qos_profile_sensor_data
        )
        self.create_subscription(
            Image, f"/{self.robot_id}/camera", self.on_image, qos_profile_sensor_data
        )

        # GLOBAL CHAT
        self.chat_pub = self.create_publisher(String, "/global_chat", 10)
        self.create_subscription(String, "/global_chat", self.on_chat, 10)

        # отправка раз в 5 сек
        self.create_timer(5.0, self.send_chat)

        # UWB / IMU
        self.uwb_count = 0
        self.imu_count = 0

        self.create_subscription(PointStamped, f"/{self.robot_id}/uwb", self.on_uwb, 10)
        self.create_subscription(IMUmsg, f"/{self.robot_id}/imu", self.on_imu, 10)

        # сервисы
        self.motor = self.create_client(MotorSRV, f"/{self.robot_id}/motor")
        self.servo = self.create_client(ServSRV, f"/{self.robot_id}/servo")

        # старт
        self.timer_start = self.create_timer(3.0, self.start_demo)

    # камера
    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            out = self.bridge.cv2_to_imgmsg(gray, "bgr8")
            self.pub.publish(out)
        except:
            pass  # игнорируем при shutdown

    # чат
    def on_chat(self, msg):
        if not msg.data.startswith(self.robot_id):
            self._log(f"CHAT: {msg.data}")

    def send_chat(self):
        msg = String()
        msg.data = f"{self.robot_id}: hello"
        self.chat_pub.publish(msg)

    # UWB
    def on_uwb(self, msg):
        if self.uwb_count < 3:
            self.uwb_count += 1
            self._log(f"UWB: {msg.point.x:.2f}, {msg.point.y:.2f}")

    # IMU
    def on_imu(self, msg):
        if self.imu_count < 3:
            self.imu_count += 1
            self._log(f"IMU: {msg.course}")

    # движение
    def start_demo(self):
        self.timer_start.cancel()

        self._log("GO")

        req = ServSRV.Request()
        req.angle = 36
        self.servo.call_async(req)

        req = MotorSRV.Request()
        req.command = "forward"
        req.speed = 50
        self.motor.call_async(req)

        self.timer_stop = self.create_timer(2.0, self.stop)

    def stop(self):
        self.timer_stop.cancel()

        req = MotorSRV.Request()
        req.command = "stop"
        req.speed = 0
        self.motor.call_async(req)

        self._log("STOP")

    def _log(self, text):
        super()._log(text)


def main():
    rclpy.init()
    node = UserNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
