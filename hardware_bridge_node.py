#!/usr/bin/env python3
"""
hardware_bridge_node.py
Node ini menjembatani ROS2 dengan perangkat keras nyata (PCA9685).
Menerima sudut (radian) dari /joint_states dan memutar servo fisik.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

# Import library hardware Adafruit
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

class HardwareBridgeNode(Node):
    def __init__(self):
        super().__init__('hardware_bridge_node')
        
        # Inisialisasi I2C dan PCA9685
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50 # Standar frekuensi servo RC adalah 50Hz
        
        # Mapping Channel PCA9685 ke Joint Robot
        # Sesuaikan dengan colokan di perangkat keras Anda
        # Format: { 'joint_index': (pca_channel, min_pulse, max_pulse, offset_derajat) }
        # min/max_pulse standar servo adalah 500-2500, tapi MG996R kadang butuh 600-2400.
        self.servos = {
            0: servo.Servo(self.pca.channels[0], min_pulse=600, max_pulse=2400), # Base
            1: servo.Servo(self.pca.channels[1], min_pulse=600, max_pulse=2400), # Shoulder
            2: servo.Servo(self.pca.channels[2], min_pulse=600, max_pulse=2400), # Elbow
        }
        
        # Channel khusus Gripper (Capit)
        self.gripper_channel = 3
        self.gripper_servo = servo.Servo(self.pca.channels[self.gripper_channel], min_pulse=600, max_pulse=2400)
        
        # Subscribers
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.cmd_sub = self.create_subscription(String, '/robot_command', self.cmd_cb, 10)
        
        self.get_logger().info("Hardware Bridge PCA9685 Aktif. Siap memutar servo!")

    def joint_cb(self, msg):
        """Menerima array Radian dari IBVS Controller dan mengubahnya ke Derajat."""
        positions_rad = msg.position
        
        for i, rad in enumerate(positions_rad):
            if i in self.servos:
                # Konversi: 0 radian = 90 derajat (Tengah)
                # Rumus: (rad * 180 / pi) + 90
                deg = math.degrees(rad) + 90.0
                
                # Batasi (Clamp) agar tidak merusak servo mekanis
                deg = max(0.0, min(180.0, deg))
                
                try:
                    self.servos[i].angle = deg
                except Exception as e:
                    self.get_logger().error(f"Error memutar servo {i}: {e}")

    def cmd_cb(self, msg):
        """Mendengarkan perintah capit."""
        cmd = msg.data
        if cmd == "ATTACH_OBJECT":
            # Tutup capit (Sesuaikan sudutnya dengan mekanik capit Anda)
            self.gripper_servo.angle = 120.0 
            self.get_logger().info("Gripper: DITUTUP")
        elif cmd == "DETACH_OBJECT":
            # Buka capit
            self.gripper_servo.angle = 45.0
            self.get_logger().info("Gripper: DIBUKA")

    def destroy_node(self):
        """Matikan semua servo (Release torque) saat node dimatikan."""
        for i in self.servos:
            self.servos[i].angle = None
        self.gripper_servo.angle = None
        self.pca.deinit()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HardwareBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()