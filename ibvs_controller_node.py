import time
import math
import numpy as np
import threading

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Point
    from std_msgs.msg import String, Float64MultiArray
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: rclpy not found. Running in standalone mode.")
    class Node: pass
    class Point:
        def __init__(self): self.x = 0.0; self.y = 0.0; self.z = 0.0
    class String:
        def __init__(self): self.data = ""
    class Float64MultiArray:
        def __init__(self): self.data = []
    class JointState:
        def __init__(self): self.position = []

class IBVSControllerNode(Node):
    def __init__(self):
        self.is_mock = not ROS2_AVAILABLE
        
        self.state = "IDLE"
        self.bbox_x = -1.0
        self.bbox_y = -1.0
        self.bbox_area = 0.0
        self.wait_ticks = 0
        
        self.container_x = 0.0
        self.container_y = -0.6
        
        self.dof = 6
        # Joint targets [Base, Shoulder, Elbow]
        self.joint_targets = [0.0]*6
        self.joint_currents = [0.0]*6
        
        # PBVS Controller variables
        self.search_dir = 1
        self.search_shoulder_dir = -1
        self.lost_ticks = 0
        self.limits_enabled = True
        self.dh_params = []
        
        if not self.is_mock:
            super().__init__('ibvs_controller_node')
            self.bbox_sub = self.create_subscription(Point, '/simulated_yolo_bbox', self.bbox_cb, 10)
            self.cmd_sub = self.create_subscription(String, '/robot_command', self.cmd_cb, 10)
            self.cmd_pub = self.create_publisher(String, '/robot_command', 10)
            self.dh_sub = self.create_subscription(Float64MultiArray, '/dh_parameters', self.dh_cb, 10)
            self.container_sub = self.create_subscription(Point, '/container_point', self.container_cb, 10)
            self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
            self.status_pub = self.create_publisher(String, '/robot_status', 10)
            self.target_pub = self.create_publisher(Point, '/target_ik_point', 10)
            self.timer = self.create_timer(0.02, self.control_loop) # 50Hz
        else:
            self.start_mock_timer()

    def bbox_cb(self, msg):
        self.bbox_x = msg.x
        self.bbox_y = msg.y
        self.bbox_area = msg.z

    def container_cb(self, msg):
        self.container_x = msg.x
        self.container_y = msg.y

    def dh_cb(self, msg):
        new_dof = len(msg.data) // 4
        self.dh_params = []
        for i in range(new_dof):
            self.dh_params.append([
                msg.data[i*4], 
                msg.data[i*4+1], 
                msg.data[i*4+2], 
                msg.data[i*4+3]
            ])
            
        if new_dof != self.dof and new_dof > 0:
            self.dof = new_dof
            while len(self.joint_targets) < self.dof:
                self.joint_targets.append(0.0)
                self.joint_currents.append(0.0)
            self.joint_targets = self.joint_targets[:self.dof]
            self.joint_currents = self.joint_currents[:self.dof]
            # Reset stale snapshot when arm geometry changes
            self.target_obj_x = 0.0
            self.target_obj_y = 0.0
            if self.state != "IDLE":
                self.state = "HOME_RETURN"
                self.wait_ticks = 0

    def cmd_cb(self, msg):
        cmd = msg.data
        if cmd == "LIMITS_ON":
            self.limits_enabled = True
            if not self.is_mock: self.get_logger().info("Mechanical limits ENABLED.")
        elif cmd == "LIMITS_OFF":
            self.limits_enabled = False
            if not self.is_mock: self.get_logger().info("Mechanical limits DISABLED.")
        elif cmd == "HOME":
            self.state = "HOME_RETURN"
            self.wait_ticks = 0
            if not self.is_mock:
                self.get_logger().info("Resetting to HOME")
        elif cmd == "START":
            self.state = "SEARCH"
            if not self.is_mock:
                self.get_logger().info("Starting IBVS Tracker")
        elif cmd == "E-STOP" or cmd == "STOP":
            self.state = "HOME_RETURN"
            self.wait_ticks = 0
            if not self.is_mock:
                self.get_logger().info("Emergency Stop! Returning HOME.")

    def calculate_fk(self, joints):
        T = np.eye(4)
        if not self.dh_params or len(self.dh_params) != self.dof:
            return T # Safe default if DH table not yet received
            
        for i in range(self.dof):
            a, alpha, d, theta_off = self.dh_params[i]
            theta = joints[i] + theta_off
            
            ct = math.cos(theta)
            st = math.sin(theta)
            ca = math.cos(alpha)
            sa = math.sin(alpha)
            
            A = np.array([
                [ct, -st*ca,  st*sa, a*ct],
                [st,  ct*ca, -ct*sa, a*st],
                [0,   sa,     ca,    d],
                [0,   0,      0,     1]
            ])
            T = np.dot(T, A)
        return T

    def calculate_all_fk(self, joints):
        matrices = []
        T = np.eye(4)
        n = min(len(self.dh_params), self.dof, len(joints))
        if n == 0:
            return matrices
            
        for i in range(n):
            a, alpha, d, theta_off = self.dh_params[i]
            theta = joints[i] + theta_off
            ct, st = math.cos(theta), math.sin(theta)
            ca, sa = math.cos(alpha), math.sin(alpha)
            A = np.array([
                [ct, -st*ca,  st*sa, a*ct],
                [st,  ct*ca, -ct*sa, a*st],
                [0,   sa,     ca,    d],
                [0,   0,      0,     1]
            ])
            T = np.dot(T, A)
            matrices.append(T)
        return matrices

    def calculate_analytical_ik(self, target_x, target_y, target_z):
        """Analytical IK for any DOF configuration (1-6).
        Always returns a list of exactly self.dof joint angle targets."""
        n = len(self.dh_params)
        if n == 0 or self.dof == 0:
            return None

        # ── Read DH params with per-index fallbacks ──
        Z_shoulder   = self.dh_params[0][2] if n >= 1 else 0.3
        base_off     = self.dh_params[0][3] if n >= 1 else 0.0
        L1           = self.dh_params[1][0] if n >= 2 else 0.4
        shoulder_off = self.dh_params[1][3] if n >= 2 else math.radians(45.0)
        L2_base      = self.dh_params[2][0] if n >= 3 else 0.3
        elbow_off    = self.dh_params[2][3] if n >= 3 else math.radians(-135.0)
        # DOF 3-5: canonical pure-rotation wrists (d=0) → no wrist extension
        # DOF 6:   full wrist with d[4]=wrist_ext, d[5]=camera lateral offset
        if self.dof >= 6 and n >= 6:
            wrist_ext      = self.dh_params[4][2]
            lateral_offset = -self.dh_params[5][2]
        else:
            wrist_ext      = 0.0
            lateral_offset = 0.0

        Rc = math.hypot(target_x, target_y)
        raw_base = math.atan2(target_y, target_x)

        # ── DOF = 1: base rotation only ──
        if self.dof == 1:
            return [raw_base - base_off]

        # ── DOF = 2: base + single-link shoulder ──
        if self.dof == 2:
            delta_z = target_z - Z_shoulder
            theta1 = math.atan2(delta_z, max(Rc, 0.001)) - shoulder_off
            return [raw_base - base_off, theta1]

        # ── DOF >= 3: full planar 2-link IK (3R with base) ──
        R_arm = math.sqrt(max(0.01, Rc**2 - lateral_offset**2))
        angle = raw_base - math.atan2(lateral_offset, R_arm) - base_off

        delta_z = target_z - Z_shoulder
        D_sq = R_arm**2 + delta_z**2
        D = math.sqrt(D_sq)
        L2 = L2_base + wrist_ext

        # Clamp to reachable workspace (outer)
        if D > L1 + L2:
            scale = (L1 + L2 - 0.001) / D
            R_arm *= scale; delta_z *= scale
            D_sq = R_arm**2 + delta_z**2

        # Clamp to reachable workspace (inner singularity)
        if D < abs(L1 - L2):
            scale = (abs(L1 - L2) + 0.001) / max(0.001, D)
            R_arm *= scale; delta_z *= scale
            D_sq = R_arm**2 + delta_z**2

        D = math.sqrt(D_sq)
        gamma    = math.acos(max(-1.0, min(1.0, (L1**2 + L2**2 - D_sq) / (2 * L1 * L2))))
        theta_B  = -(math.pi - gamma)
        phi      = math.atan2(delta_z, R_arm)
        alpha_ik = math.acos(max(-1.0, min(1.0, (L1**2 + D_sq - L2**2) / (2 * L1 * D))))
        theta_A  = phi + alpha_ik

        # Build result: first 3 joints from IK, remaining wrist joints at 0
        result = [angle, theta_A - shoulder_off, theta_B - elbow_off]
        while len(result) < self.dof:
            result.append(0.0)
        return result[:self.dof]

    def control_loop(self):
        if self.state == "IDLE":
            # Don't publish or spam CPU when idling or resting at home
            return
            
        prev_targets = list(self.joint_targets)
            
        if self.state == "SEARCH":
            # Dynamic continuous sweep without teleportation jerks
            if self.dof > 0:
                self.joint_targets[0] += 0.015 * self.search_dir
                if self.joint_targets[0] > 1.2: self.search_dir = -1
                if self.joint_targets[0] < -1.2: self.search_dir = 1
                
            # Sweep shoulder up and down to scan various radii dynamically
            if self.dof > 1:
                self.joint_targets[1] += 0.008 * self.search_shoulder_dir
                if self.joint_targets[1] > 0.2: self.search_shoulder_dir = -1
                if self.joint_targets[1] < -0.8: self.search_shoulder_dir = 1
            
            # If object found
            if self.bbox_x >= 0:
                # INSTANT PBVS SNAPSHOT
                u = self.bbox_x * 320.0
                v = self.bbox_y * 240.0
                cx_ratio = (u - 160) / 250.0  # focal_length = 250
                cy_ratio = (v - 120) / 250.0
                
                # ray in local camera frame: [X, Y, Z] (Camera optical axis is +X)
                ray_local = np.array([1.0, cy_ratio, -cx_ratio, 0.0])
                
                T_tcp = self.calculate_fk(self.joint_targets)
                ray_global = np.dot(T_tcp, ray_local)
                
                P_tcp = T_tcp[:3, 3]
                X_vec = ray_global[:3] 
                
                if X_vec[2] == 0:
                    lambda_val = 0.0
                else:
                    lambda_val = -P_tcp[2] / X_vec[2] # Floor intersection at Z=0
                
                self.target_obj_x = P_tcp[0] + lambda_val * X_vec[0]
                self.target_obj_y = P_tcp[1] + lambda_val * X_vec[1]

                # Publish locked target so viz can show accurate dot immediately
                if not self.is_mock:
                    pt = Point()
                    pt.x = float(self.target_obj_x)
                    pt.y = float(self.target_obj_y)
                    pt.z = 0.0
                    self.target_pub.publish(pt)

                self.state = "DIVING"
                if not self.is_mock:
                    self.get_logger().info(f"Target Seen! Pinhole Snapshot: X={self.target_obj_x:.2f}, Y={self.target_obj_y:.2f}. DIVING.")
                else:
                    print(f"Target Seen! PBVS Snapshot: X={self.target_obj_x:.2f}, Y={self.target_obj_y:.2f}. DIVING.")
                
        elif self.state == "DIVING":
            # Using IK to directly reach the locked coordinate received from viz_env_node
            ik_solution = self.calculate_analytical_ik(self.target_obj_x, self.target_obj_y, 0.03)
            if ik_solution:
                error = 0.0
                for i, tgt in enumerate(ik_solution):
                    if i < self.dof:
                        diff = tgt - self.joint_targets[i]
                        self.joint_targets[i] += np.clip(diff, -0.015, 0.015)
                        error += abs(diff)

                if error < 0.02:
                    self.state = "GRABBING"
                    if not self.is_mock:
                        self.get_logger().info("Target Reached. GRAB_READY.")
                    else:
                        print("Target Reached. GRAB_READY.")
                
        elif self.state == "GRABBING":
            if not self.is_mock:
                msg = String(); msg.data = "ATTACH_OBJECT"
                self.cmd_pub.publish(msg)
            else:
                print("MOCK COMMAND: ATTACH_OBJECT")
                
            self.wait_ticks = 0
            self.state = "LIFTING"
            
        elif self.state == "LIFTING":
            # Use IK to move to a safe hover height above the grabbed point
            ik_solution = self.calculate_analytical_ik(self.target_obj_x, self.target_obj_y, 0.25)
            if ik_solution:
                for i, tgt in enumerate(ik_solution):
                    if i < self.dof:
                        self.joint_targets[i] += np.clip(tgt - self.joint_targets[i], -0.02, 0.02)
            self.wait_ticks += 1
            if self.wait_ticks > 50:
                self.state = "DELIVERING"
                self.wait_ticks = 0
                if not self.is_mock: self.get_logger().info("Lifted! Delivering to container.")
                
        elif self.state == "DELIVERING":
            ik_solution = self.calculate_analytical_ik(self.container_x, self.container_y, 0.1)
            if ik_solution:
                for i, tgt in enumerate(ik_solution):
                    if i < self.dof:
                        self.joint_targets[i] += np.clip(tgt - self.joint_targets[i], -0.015, 0.015)

            self.wait_ticks += 1
            if self.wait_ticks > 150:
                self.state = "DROPPING"
                self.wait_ticks = 0
                if not self.is_mock: self.get_logger().info("Arrived at container. Dropping.")
                
        elif self.state == "DROPPING":
            if self.wait_ticks == 0:
                if not self.is_mock:
                    msg = String(); msg.data = "DETACH_OBJECT"
                    self.cmd_pub.publish(msg)
                else:
                    print("MOCK COMMAND: DETACH_OBJECT")
            
            # Use IK to lift above container after drop
            ik_solution = self.calculate_analytical_ik(self.container_x, self.container_y, 0.25)
            if ik_solution:
                for i, tgt in enumerate(ik_solution):
                    if i > 0 and i < self.dof:  # skip base rotation during lift-clear
                        self.joint_targets[i] += np.clip(tgt - self.joint_targets[i], -0.02, 0.02)
            self.wait_ticks += 1
            
            if self.wait_ticks > 40:
                self.state = "HOME_RETURN"
                self.wait_ticks = 0
                if not self.is_mock:
                    self.get_logger().info("Object dropped. Returning to HOME.")
                    # Reset camera perception: publish sentinel target to flush vision
                    flush_msg = Point()
                    flush_msg.x, flush_msg.y, flush_msg.z = -9.0, -9.0, -9.0
                    self.target_pub.publish(flush_msg)

        elif self.state == "HOME_RETURN":
            error = 0.0
            for i in range(self.dof):
                diff = 0.0 - self.joint_targets[i]
                self.joint_targets[i] += np.clip(diff, -0.02, 0.02)
                error += abs(diff)
                
            if error < 0.01:
                self.state = "IDLE"
                if not self.is_mock: self.get_logger().info("Home Reached. IDLE.")
                
        # Virtual Fixture Collision Prevention (Floor Z=0)
        # Skip link [0] (robot base/pedestal - always at floor level by design)
        if self.limits_enabled and self.dof > 0:
            matrices = self.calculate_all_fk(self.joint_targets)
            violation = False
            for M in matrices[1:]:  # Skip base link
                if M[2, 3] < 0.01: # 1cm safety barrier over the floor
                    violation = True
                    break
                    
            if violation:
                # Revert modifying actions
                self.joint_targets = prev_targets
                
                # State-specific boundary handlers
                if self.state == "SEARCH":
                    self.search_shoulder_dir *= -1
                elif self.state == "DIVING":
                    self.state = "GRABBING"
                    if not self.is_mock:
                        self.get_logger().info("Floor limits engaged during dive. Forcing GRAB.")
                    else:
                        print("Floor limits engaged during dive. Forcing GRAB.")

        # Bypass old EMA filter - PBVS calculates flawless trajectories inherently
        self.joint_currents = list(self.joint_targets)

        self.publish_joints()
        self.publish_status()

    def publish_status(self):
        if self.is_mock: return
        msg = String()
        # Map internal state to UI state
        UI_state = self.state
        if self.state == "SEARCH": UI_state = "SCANNING"
        elif self.state == "CENTERING" or self.state == "DIVING": UI_state = "TRACKING"
        elif self.state == "GRABBING" or self.state == "LIFTING": UI_state = "PICKING"
        elif self.state == "DELIVERING" or self.state == "DROPPING": UI_state = "PLACING"
        elif self.state == "HOME_RETURN": UI_state = "HOMING"
        msg.data = UI_state
        self.status_pub.publish(msg)

    def publish_joints(self):
        if self.is_mock: return
        msg = JointState()
        msg.position = self.joint_currents
        self.joint_pub.publish(msg)

    def start_mock_timer(self):
        def loop():
            while True:
                self.control_loop()
                time.sleep(0.02)
        threading.Thread(target=loop, daemon=True).start()

def main():
    if ROS2_AVAILABLE:
        rclpy.init()
        
    node = IBVSControllerNode()
    
    if ROS2_AVAILABLE:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    else:
        # Just keep running in mock mode
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
