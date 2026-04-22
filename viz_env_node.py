# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import math
import queue
import random

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Point
    from std_msgs.msg import Float64MultiArray, String
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: rclpy not found. Running in standalone Tkinter mode for visual verification.")
    class Node: pass
    class Point:
        def __init__(self): self.x = 0.0; self.y = 0.0; self.z = 0.0
    class Float64MultiArray:
        def __init__(self): self.data = []
    class String:
        def __init__(self): self.data = ""
    class JointState:
        def __init__(self): self.position = []

class VizEnvNode(Node):
    def __init__(self):
        self.is_mock = not ROS2_AVAILABLE
        
        self.msg_queue = queue.Queue()
        self.target_queue = queue.Queue()
        
        if not self.is_mock:
            super().__init__('viz_env_node')
            self.target_pub = self.create_publisher(Point, '/target_ik_point', 10)
            self.dh_pub = self.create_publisher(Float64MultiArray, '/dh_parameters', 10)
            self.container_pub = self.create_publisher(Point, '/container_point', 10)
            self.bbox_sub = self.create_subscription(Point, '/simulated_yolo_bbox', self.bbox_cb, 10)

            self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
            self.target_sub = self.create_subscription(Point, '/target_ik_point', self.target_callback, 10)
            self.cmd_sub = self.create_subscription(String, '/robot_command', self.cmd_cb, 10)
            self.status_sub = self.create_subscription(String, '/robot_status', self.status_cb, 10)

            self.cmds_rcvd   = queue.Queue()
            self.bbox_queue  = queue.Queue()
            self.status_queue = queue.Queue()

    def bbox_cb(self, msg):
        self.bbox_queue.put((msg.x, msg.y, msg.z))

    def status_cb(self, msg):
        if hasattr(self, 'status_queue'):
            self.status_queue.put(msg.data)

    def cmd_cb(self, msg):
        self.cmds_rcvd.put(msg.data)

    def joint_callback(self, msg):
        self.msg_queue.put(list(msg.position))

    def target_callback(self, msg):
        self.target_queue.put((msg.x, msg.y, msg.z))

    def publish_target(self, x, y, z):
        if self.is_mock: return
        msg = Point()
        msg.x, msg.y, msg.z = float(x), float(y), float(z)
        self.target_pub.publish(msg)

    def publish_dh(self, dh_list):
        if self.is_mock: return
        msg = Float64MultiArray()
        msg.data = [float(val) for row in dh_list for val in row]
        self.dh_pub.publish(msg)

    def publish_command(self, cmd_str):
        if self.is_mock: return
        msg = String()
        msg.data = cmd_str
        # Hack to re-initialize cmd_pub since it was removed from __init__ accidentally
        if not hasattr(self, 'cmd_pub'):
            self.cmd_pub = self.create_publisher(String, '/robot_command', 10)
        self.cmd_pub.publish(msg)
        
    def publish_container(self, x, y, z):
        if self.is_mock: return
        msg = Point()
        msg.x, msg.y, msg.z = float(x), float(y), float(z)
        self.container_pub.publish(msg)

class RobotGUI:
    def __init__(self, root, ros_node):
        self.root = root
        self.node = ros_node
        self.root.title("IBVS Simulator & 6-DOF Configurator")
        self.root.geometry("1630x750")
        self.root.resizable(True, True)
        
        self.dof = 6
        self.current_joints = [0.0] * self.dof # Now clean Zeroes thanks to new DH!
        self.target_x, self.target_y, self.target_z = 0.0, 0.0, 0.0
        self.has_target       = False
        self.cam_mode         = "manual"   # "manual" | "realcam" | "workspace"
        self.cam_detecting    = False      # True = fresh YOLO detection this cycle
        self.cam_target_locked = False    # True = controller gave us authoritative position

        self.is_grabbed       = False
        self.is_homing        = False   # True while HOME_RETURN traveling
        self.gripper_open     = True
        self.container_pos = [0.0, -0.6, 0.0] # Static container at Y=-0.6m

        # ── Workspace area definition ──────────────────────────────
        self.ws_mode          = "idle"   # "idle" | "drawing" | "defined"
        self.ws_p1            = None     # (cx, cy) first corner in canvas px
        self.ws_p2            = None     # (cx, cy) second corner in canvas px
        self.ws_world_rect    = None     # (xmin, ymin, xmax, ymax) in world meters
        # ── Auto-track trigger ─────────────────────────────────────
        self.auto_track_armed = False    # waiting for detection inside area
        self.auto_track_sent  = False    # START already published this cycle
        
        self.dh_params = [
            [0.0, 90.0, 0.3, 0.0],
            [0.4, 0.0, 0.0, 45.0],
            [0.3, 0.0, 0.0, -135.0],
            [0.0, 90.0, 0.0, 90.0],
            [0.0, -90.0, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.0]
        ]
        # Canonical DH presets per DOF — verified correct via FK test
        # DOF 3-5: pure rotation wrists (d=0, a=0), camera at joint-2 tip
        # DOF 6:   full wrist + camera extension (d[4]=0.1, d[5]=0.1)
        self.CANONICAL_DH = {
            3: [[0.0, 90.0, 0.3, 0.0],
                [0.4, 0.0,  0.0, 45.0],
                [0.3, 0.0,  0.0, -135.0]],
            4: [[0.0, 90.0, 0.3, 0.0],
                [0.4, 0.0,  0.0, 45.0],
                [0.3, 0.0,  0.0, -135.0],
                [0.0, 90.0, 0.0, 90.0]],
            5: [[0.0,  90.0, 0.3, 0.0],
                [0.4,   0.0, 0.0, 45.0],
                [0.3,   0.0, 0.0, -135.0],
                [0.0,  90.0, 0.0, 90.0],
                [0.0, -90.0, 0.0, 0.0]],
            6: [[0.0,  90.0, 0.3,   0.0],
                [0.4,   0.0, 0.0,  45.0],
                [0.3,   0.0, 0.0, -135.0],
                [0.0,  90.0, 0.0,  90.0],
                [0.0, -90.0, 0.1,   0.0],
                [0.0,   0.0, 0.1,   0.0]],
        }
        self.active_dh_edit = None
        
        self.scale_factor = 150.0  # pixels per meter
        self.origin_x = 400
        self.origin_y = 165
        self.front_origin_y = 280
        
        self.setup_ui()
        self.publish_current_dh()
        self.generate_workspace()
        
        self.update_loop()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='#e0e0e0', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10, 'bold'))
        
        main_frame = ttk.Frame(self.root, padding=0)
        main_frame.grid(row=0, column=0, sticky='nsew')
        self.root.configure(bg='#1e1e1e')
        
        # LEFT FRAME (Visuals) width 800
        left_frame = ttk.Frame(main_frame, width=800, height=730)
        left_frame.grid(row=0, column=0, sticky='nsew')
        left_frame.grid_propagate(False)
        
        lbl_top = ttk.Label(left_frame, text="Top View (X-Y Projection) - Click to set (X,Y) target", font=('Helvetica', 12, 'bold'))
        lbl_top.pack(pady=(5,0))
        self.canvas_top = tk.Canvas(left_frame, width=800, height=330, bg='#2d2d2d', highlightthickness=1, highlightbackground='#3d3d3d')
        self.canvas_top.pack(pady=5)
        self.canvas_top.bind("<Button-1>", self.on_top_click)
        self.canvas_top.bind("<Button-3>", self.on_top_right_click)
        self.canvas_top.bind("<Motion>", self.on_top_hover)
        self.canvas_top.bind("<Leave>", self.on_hover_leave)
        
        lbl_bottom = ttk.Label(left_frame, text="Front View (X-Z Projection) - Click to set (X,Z) target", font=('Helvetica', 12, 'bold'))
        lbl_bottom.pack(pady=(5,0))
        self.canvas_bottom = tk.Canvas(left_frame, width=800, height=330, bg='#2d2d2d', highlightthickness=1, highlightbackground='#3d3d3d')
        self.canvas_bottom.pack(pady=5)
        self.canvas_bottom.bind("<Button-1>", self.on_front_click)
        self.canvas_bottom.bind("<Button-3>", self.on_front_right_click)
        self.canvas_bottom.bind("<Motion>", self.on_front_hover)
        self.canvas_bottom.bind("<Leave>", self.on_hover_leave)
        
        # MID FRAME (Configurator) width 420
        mid_frame = ttk.Frame(main_frame, width=420, height=730)
        mid_frame.grid(row=0, column=1, sticky='nsew', padx=10)
        mid_frame.grid_propagate(False)
        
        dof_frame = ttk.LabelFrame(mid_frame, text="Robot Configuration", padding=10)
        dof_frame.pack(fill='x', pady=5)
        ttk.Label(dof_frame, text="Degrees of Freedom (DOF):").grid(row=0, column=0, padx=5, pady=5)
        self.dof_var = tk.StringVar(value=str(self.dof))
        self.dof_combo = ttk.Combobox(dof_frame, textvariable=self.dof_var,
                                      values=[str(i) for i in range(3, 7)],
                                      width=5, state="readonly")
        self.dof_combo.grid(row=0, column=1, padx=5, pady=5)
        self.dof_combo.bind("<<ComboboxSelected>>", self.on_dof_change)

        self.dh_frame = ttk.LabelFrame(mid_frame, text="DH Parameters (a, alpha, d, theta_offset)", padding=10)
        self.dh_frame.pack(fill='x', pady=5)
        
        self.js_frame = ttk.LabelFrame(mid_frame, text="Joint States (Current θ)", padding=10)
        self.js_frame.pack(fill='both', pady=5, expand=True)
        
        self.build_dh_and_js_panels()
        
        # RIGHT FRAME (IBVS & Controls) width 370
        right_frame = ttk.Frame(main_frame, width=370, height=730)
        right_frame.grid(row=0, column=2, sticky='nsew', padx=10)
        right_frame.grid_propagate(False)
        
        ttk.Label(right_frame, text="Camera View (PIP)", font=('Helvetica', 12, 'bold')).pack(pady=(10, 5))
        self.pip_canvas = tk.Canvas(right_frame, width=320, height=240, bg="#000000", highlightthickness=2, highlightbackground="#00FF00")
        self.pip_canvas.pack()

        # ── Target Input Mode toggle ───────────────────────────────
        mode_frame = ttk.LabelFrame(right_frame, text="Target Input Mode", padding=6)
        mode_frame.pack(fill='x', pady=(8, 0))
        self.cam_mode_var = tk.StringVar(value="manual")
        ttk.Radiobutton(mode_frame, text="Manual (Click on canvas)",
                        variable=self.cam_mode_var, value="manual",
                        command=self._on_mode_change).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Real Camera (Auto-track)",
                        variable=self.cam_mode_var, value="realcam",
                        command=self._on_mode_change).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Define Workspace Area  ✚",
                        variable=self.cam_mode_var, value="workspace",
                        command=self._on_mode_change).pack(anchor='w')
        self.lbl_cam_status = ttk.Label(mode_frame,
                                        text="Mode: MANUAL — click canvas to set target",
                                        foreground="#ffff00", font=("Courier", 9))
        self.lbl_cam_status.pack(anchor='w', pady=(2, 0))

        # ── Workspace Area Panel ─────────────────────────────────────
        ws_frame = ttk.LabelFrame(right_frame, text="Workspace Area Control", padding=6)
        ws_frame.pack(fill='x', pady=(6, 0))

        self.lbl_ws_status = ttk.Label(ws_frame,
                                       text="⬜ Area: Not defined",
                                       foreground="#888888", font=("Courier", 9),
                                       wraplength=340, justify='left')
        self.lbl_ws_status.pack(anchor='w', pady=(0, 4))

        self.btn_arm = ttk.Button(ws_frame, text="ARM AUTO-TRACK",
                                  command=self._arm_auto_track, state='disabled')
        self.btn_arm.pack(fill='x', pady=(0, 2))

        ttk.Button(ws_frame, text="CLEAR WORKSPACE",
                   command=self._clear_workspace).pack(fill='x', pady=(0, 4))

        rearm_frame = ttk.LabelFrame(ws_frame, text="Re-Arm After Pick Cycle", padding=4)
        rearm_frame.pack(fill='x', pady=(2, 0))
        self.rearm_mode = tk.StringVar(value="manual")
        ttk.Radiobutton(rearm_frame, text="Manual  — click ARM again",
                        variable=self.rearm_mode, value="manual").pack(anchor='w')
        ttk.Radiobutton(rearm_frame, text="Auto-Loop — re-arm on IDLE",
                        variable=self.rearm_mode, value="autoloop").pack(anchor='w')

        # YOLO BBox Tracker telemetry (kept below mode toggle)
        ttk.Label(right_frame, text="YOLO BBox Tracker", font=('Helvetica', 12, 'bold')).pack(pady=(8, 5))
        self.lbl_telemetry = ttk.Label(right_frame, text="X: 0.0 | Y: 0.0 | Area: 0.0", font=('Courier', 11, 'bold'), background="#1e1e1e", foreground="#00ffcc")
        self.lbl_telemetry.pack()

        
        # Task
        task_frame = ttk.LabelFrame(right_frame, text="Manual Target (X, Y, Z) - m", padding=10)
        task_frame.pack(fill='x', pady=20)
        
        self.var_x = tk.StringVar(value="0.0")
        self.var_y = tk.StringVar(value="0.0")
        self.var_z = tk.StringVar(value="0.0")
        
        ttk.Label(task_frame, text="X:").grid(row=0, column=0)
        ttk.Entry(task_frame, textvariable=self.var_x, width=6).grid(row=0, column=1)
        ttk.Label(task_frame, text="Y:").grid(row=0, column=2)
        ttk.Entry(task_frame, textvariable=self.var_y, width=6).grid(row=0, column=3)
        ttk.Label(task_frame, text="Z (Lantai):").grid(row=0, column=4)
        ttk.Entry(task_frame, textvariable=self.var_z, width=6, state="disabled").grid(row=0, column=5)
        ttk.Button(task_frame, text="SEND", command=self.send_manual_target).grid(row=1, column=0, columnspan=6, pady=5, sticky='we')

        # Commands
        ctrl_frame = ttk.LabelFrame(right_frame, text="Commands (IBVS & General)", padding=10)
        ctrl_frame.pack(fill='x', pady=5)
        
        self.limits_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Enable Limits", variable=self.limits_enabled, command=self.toggle_limits).pack(fill='x', pady=2)
        
        ttk.Button(ctrl_frame, text="START TRACKING (IBVS)", command=lambda: self.node.publish_command("START")).pack(fill='x', pady=5)
        ttk.Button(ctrl_frame, text="E-STOP", command=lambda: self.node.publish_command("E-STOP")).pack(fill='x', pady=5)
        ttk.Button(ctrl_frame, text="HOME", command=lambda: self.node.publish_command("HOME")).pack(fill='x', pady=5)

    def build_dh_and_js_panels(self):
        for widget in self.dh_frame.winfo_children():
            widget.destroy()
        
        self.dh_vars = []
        headers = ['Link', 'a (m)', 'alpha (deg)', 'd (m)', 'theta_offset']
        for col, h in enumerate(headers):
            ttk.Label(self.dh_frame, text=h, font=('Helvetica', 8, 'bold')).grid(row=0, column=col, padx=2, pady=2)
        
        for i in range(self.dof):
            ttk.Label(self.dh_frame, text=f"L{i+1}").grid(row=i+1, column=0, padx=2, pady=2)
            row_vars = []
            for j in range(4):
                var = tk.StringVar(value=str(self.dh_params[i][j]))
                
                if j == 0 or j == 2:
                    ent = ttk.Spinbox(self.dh_frame, textvariable=var, width=7, from_=-2.0, to=2.0, increment=0.05)
                else:
                    ent = ttk.Spinbox(self.dh_frame, textvariable=var, width=7, from_=-360.0, to=360.0, increment=5.0)
                    
                ent.grid(row=i+1, column=j+1, padx=2, pady=2)
                
                ent.bind('<Return>', lambda e, rv=row_vars, idx=i: self.update_dh_from_ui(idx, rv))
                ent.bind('<FocusIn>', lambda e, r=i, c=j, v=var: self.on_dh_focus(r, c, v))
                ent.config(command=lambda rv=row_vars, idx=i: self.update_dh_from_ui(idx, rv))
                
                row_vars.append(var)
            self.dh_vars.append(row_vars)
            
        master_frame = ttk.LabelFrame(self.dh_frame, text="Master DH Slider (Live Tweak)")
        master_frame.grid(row=self.dof+1, column=0, columnspan=5, pady=5, sticky='we')
        self.lbl_master_slider = ttk.Label(master_frame, text="Click a DH parameter above to begin tweaking")
        self.lbl_master_slider.pack(side=tk.TOP)
        self.master_dh_slider = tk.Scale(master_frame, from_=-180, to=180, orient="horizontal", command=self.on_master_slider_moved, state='disabled')
        self.master_dh_slider.pack(fill=tk.X, expand=True)
        self.master_dh_slider.bind("<ButtonRelease-1>", lambda e: self.publish_current_dh())
            
        ttk.Button(self.dh_frame, text="UPDATE DH", command=self.update_dh).grid(row=self.dof+2, column=0, columnspan=5, pady=10)

        for widget in self.js_frame.winfo_children():
            widget.destroy()
            
        self.lbl_joints = []
        for i in range(self.dof):
            lbl = ttk.Label(self.js_frame, text=f"θ{i+1}: 0.00°", font=('Courier', 11))
            lbl.grid(row=i//2, column=i%2, padx=10, pady=2, sticky='w')
            self.lbl_joints.append(lbl)
            
        self.lbl_tcp = ttk.Label(self.js_frame, text="TCP: X=0.00 Y=0.00 Z=0.00", font=('Courier', 10, 'bold'), foreground='#00ffcc')
        self.lbl_tcp.grid(row=(self.dof//2)+1, column=0, columnspan=2, pady=10)

    def on_dof_change(self, event):
        new_dof = int(self.dof_var.get())
        if new_dof == self.dof: return

        # Apply canonical preset for the selected DOF
        self.dh_params = [list(row) for row in self.CANONICAL_DH[new_dof]]
        self.dof = new_dof
        self.current_joints = [0.0] * self.dof

        self.build_dh_and_js_panels()
        self.publish_current_dh()
        self.generate_workspace()
        self.draw_robot()

    def on_dh_focus(self, r, c, var):
        self.active_dh_edit = (r, c, var)
        labels = ["Radius (a) [m]", "Alpha [deg]", "Depth (d) [m]", "Theta Offset [deg]"]
        self.lbl_master_slider.config(text=f"Live Edit: Link {r} | {labels[c]}")
        self.master_dh_slider.config(state='normal')
        
        if c == 0 or c == 2:
            self.master_dh_slider.config(from_=-2.0, to=2.0, resolution=0.01)
        else:
            self.master_dh_slider.config(from_=-360.0, to=360.0, resolution=1.0)
            
        try:
            val = float(var.get())
            self.master_dh_slider.set(val)
        except ValueError: pass

    def on_master_slider_moved(self, val):
        if not self.active_dh_edit: return
        r, c, var = self.active_dh_edit
        var.set(f"{float(val):.3f}")
        try:
            self.dh_params[r][c] = float(val)
            self.draw_robot()
        except ValueError: pass

    def update_dh_from_ui(self, i, vars_row):
        try:
            for j in range(4):
                self.dh_params[i][j] = float(vars_row[j].get())
            self.draw_robot()
            self.publish_current_dh()
            
            if self.active_dh_edit and self.active_dh_edit[0] == i:
                try: 
                    self.master_dh_slider.set(float(self.active_dh_edit[2].get()))
                except ValueError: pass
        except ValueError:
            pass

    def publish_current_dh(self):
        rad_params = []
        for row in self.dh_params:
            rad_params.append([row[0], math.radians(row[1]), row[2], math.radians(row[3])])
        self.node.publish_dh(rad_params)

    def update_dh(self):
        try:
            for i in range(self.dof):
                for j in range(4):
                    self.dh_params[i][j] = float(self.dh_vars[i][j].get())
            self.publish_current_dh()
            self.generate_workspace()
            self.draw_robot()
        except ValueError:
            print("Invalid DH parameter input!")

    def send_manual_target(self):
        try:
            self.target_x = float(self.var_x.get())
            self.target_y = float(self.var_y.get())
            self.target_z = float(self.var_z.get())
            self.has_target = True
            
            self.node.publish_target(self.target_x, self.target_y, self.target_z)
            self.draw_robot()
        except ValueError:
            print("Invalid target coordinates!")

    def toggle_limits(self):
        if self.limits_enabled.get():
            self.node.publish_command("LIMITS_ON")
        else:
            self.node.publish_command("LIMITS_OFF")

    def _on_mode_change(self):
        self.cam_mode = self.cam_mode_var.get()
        if self.cam_mode == "realcam":
            self.lbl_cam_status.config(
                text="Mode: REAL CAM — bbox auto-tracks canvas",
                foreground="#00ffcc")
            self.canvas_top.config(cursor="")
        elif self.cam_mode == "workspace":
            self.lbl_cam_status.config(
                text="Mode: WORKSPACE — click 2 pts on Top View",
                foreground="#FF8C00")
            self.canvas_top.config(cursor="crosshair")
            # Reset drawing state when re-entering mode mid-draw
            if self.ws_mode == "drawing":
                self.ws_mode = "idle"
                self.ws_p1 = None
                self.canvas_top.delete("ws_preview")
                self._update_workspace_ui()
        else:
            self.lbl_cam_status.config(
                text="Mode: MANUAL — click canvas to set target",
                foreground="#ffff00")
            self.canvas_top.config(cursor="")

    def ray_cast_to_floor(self, norm_x, norm_y):
        """Cast ray from camera through (norm_x, norm_y) pixel → floor Z=0.
        Returns (world_x, world_y) or None if ray doesn't reach floor."""
        fx = 250.0          # virtual focal length — must match ibvs_controller
        u  = norm_x * 320.0
        v  = norm_y * 240.0
        cx_r = (u - 160.0) / fx
        cy_r = (v - 120.0) / fx

        # Camera convention: optical = +X_cam, img-right = -Z_cam, img-down = +Y_cam
        ray_cam = np.array([1.0, cy_r, -cx_r, 0.0])

        _, Ts   = self.calculate_fk(self.current_joints)
        T_ee    = Ts[-1]
        ray_world = T_ee @ ray_cam
        P_tcp   = T_ee[:3, 3]
        X_vec   = ray_world[:3]

        # ── Primary: ray→floor intersection (Z = 0) ───────────────
        if abs(X_vec[2]) > 1e-3 and (-P_tcp[2] / X_vec[2]) >= 0:
            lam = -P_tcp[2] / X_vec[2]
            return P_tcp[0] + lam * X_vec[0], P_tcp[1] + lam * X_vec[1]

        # ── Fallback: camera not facing floor (e.g. home position) ─
        # Project onto a plane perpendicular to optical axis at assumed depth R.
        # R is chosen as the camera height (or 0.5 m minimum).
        R = max(abs(P_tcp[2]), 0.40)   # assumed look-distance
        # camera right = -Z column of T_ee, camera DOWN = +Y column
        optical = T_ee[:3, 0]
        right   = -T_ee[:3, 2]        # image-right direction in world
        down    = T_ee[:3, 1]         # image-down  direction in world

        # Project hit-point onto floor Z=0 along vertical
        hit = P_tcp + R * optical + cx_r * R * right + cy_r * R * down
        wx  = hit[0]
        wy  = hit[1]
        # Z of hit might not be 0; clamp visually (we only draw floor-plane)
        return wx, wy


    def on_top_click(self, event):
        if self.cam_mode == "workspace":
            self._handle_workspace_click(event)
            return
        if self.cam_mode != "manual":
            return
        tx = (event.x - self.origin_x) / self.scale_factor
        ty = -(event.y - self.origin_y) / self.scale_factor
        
        max_reach = sum(abs(row[0]) for row in self.dh_params)
        dist = math.sqrt(tx*tx + ty*ty)
        if dist > max_reach:
            ratio = max_reach / dist
            tx *= ratio
            ty *= ratio
            
        self.target_x = tx
        self.target_y = ty
        self.target_z = 0.0 # Force Floor Z=0
        
        self.has_target = True
        self.var_x.set(f"{self.target_x:.3f}")
        self.var_y.set(f"{self.target_y:.3f}")
        self.var_z.set("0.0")
        
        self.node.publish_target(self.target_x, self.target_y, self.target_z)
        self.draw_robot()

    def on_front_click(self, event):
        if self.cam_mode != "manual":
            return
        tx = (event.x - self.origin_x) / self.scale_factor
        ty = self.target_y
        
        max_reach = sum(abs(row[0]) for row in self.dh_params)
        dist = math.sqrt(tx*tx + ty*ty)
        if dist > max_reach:
            ratio = max_reach / dist
            tx *= ratio
            ty *= ratio
            
        self.target_x = tx
        # ignore Z axis changes
        self.target_z = 0.0
        
        self.has_target = True
        self.var_x.set(f"{self.target_x:.3f}")
        self.var_z.set("0.0")
        
        self.node.publish_target(self.target_x, self.target_y, self.target_z)
        self.draw_robot()

    def on_top_right_click(self, event):
        tx = (event.x - self.origin_x) / self.scale_factor
        ty = -(event.y - self.origin_y) / self.scale_factor
        self.container_pos = [tx, ty, 0.0]
        self.node.publish_container(tx, ty, 0.0)
        self.draw_robot()

    def on_front_right_click(self, event):
        tx = (event.x - self.origin_x) / self.scale_factor
        self.container_pos[0] = tx
        self.node.publish_container(self.container_pos[0], self.container_pos[1], 0.0)
        self.draw_robot()

    def on_top_hover(self, event):
        x = (event.x - self.origin_x) / self.scale_factor
        y = -(event.y - self.origin_y) / self.scale_factor
        self.canvas_top.delete("hover")
        self.canvas_top.create_text(event.x + 15, event.y - 15, text=f"X:{x:.2f}, Y:{y:.2f}", fill="#00ffcc", font=("Courier", 11, "bold"), tags="hover", anchor="sw")
        if self.cam_mode == "workspace":
            self._handle_workspace_motion(event)

    def on_front_hover(self, event):
        x = (event.x - self.origin_x) / self.scale_factor
        z = -(event.y - self.front_origin_y) / self.scale_factor
        self.canvas_bottom.delete("hover")
        self.canvas_bottom.create_text(event.x + 15, event.y - 15, text=f"X:{x:.2f}, Z:{z:.2f}", fill="#00ffcc", font=("Courier", 11, "bold"), tags="hover", anchor="sw")

    def on_hover_leave(self, event):
        self.canvas_top.delete("hover")
        self.canvas_bottom.delete("hover")
        if self.cam_mode == "workspace":
            self.canvas_top.delete("ws_preview")

    def _handle_workspace_click(self, event):
        """Handle click in workspace-define mode: first click = p1, second = p2."""
        if self.ws_mode in ("idle", "defined"):
            # Start a new rectangle
            self.ws_p1    = (event.x, event.y)
            self.ws_p2    = None
            self.ws_mode  = "drawing"
            self.ws_world_rect    = None
            self.auto_track_armed = False
            self.auto_track_sent  = False
            self.canvas_top.delete("ws_rect")
            self._update_workspace_ui()
        elif self.ws_mode == "drawing":
            # Complete the rectangle
            self.ws_p2   = (event.x, event.y)
            self.ws_mode = "defined"
            self.canvas_top.delete("ws_preview")
            # Convert canvas coords → world meters (canvas Y inverted vs world Y)
            x_min_px = min(self.ws_p1[0], self.ws_p2[0])
            x_max_px = max(self.ws_p1[0], self.ws_p2[0])
            y_min_px = min(self.ws_p1[1], self.ws_p2[1])
            y_max_px = max(self.ws_p1[1], self.ws_p2[1])
            xw_min = (x_min_px - self.origin_x) / self.scale_factor
            xw_max = (x_max_px - self.origin_x) / self.scale_factor
            yw_min = -(y_max_px - self.origin_y) / self.scale_factor
            yw_max = -(y_min_px - self.origin_y) / self.scale_factor
            self.ws_world_rect = (xw_min, yw_min, xw_max, yw_max)
            self._update_workspace_ui()
            self.draw_robot()

    def _handle_workspace_motion(self, event):
        """Live rectangle preview while user is drawing workspace area."""
        if self.ws_mode == "drawing" and self.ws_p1:
            self.canvas_top.delete("ws_preview")
            self.canvas_top.create_rectangle(
                self.ws_p1[0], self.ws_p1[1], event.x, event.y,
                outline="#FF8C00", width=2, dash=(6, 3), tags="ws_preview"
            )
            # Show live size label in centre of rectangle
            x1w = (min(self.ws_p1[0], event.x) - self.origin_x) / self.scale_factor
            x2w = (max(self.ws_p1[0], event.x) - self.origin_x) / self.scale_factor
            y1w = -(max(self.ws_p1[1], event.y) - self.origin_y) / self.scale_factor
            y2w = -(min(self.ws_p1[1], event.y) - self.origin_y) / self.scale_factor
            lbl = f"\u0394X={abs(x2w-x1w):.2f}m  \u0394Y={abs(y2w-y1w):.2f}m"
            mx = (self.ws_p1[0] + event.x) / 2
            my = (self.ws_p1[1] + event.y) / 2
            self.canvas_top.create_text(mx, my, text=lbl,
                fill="#FF8C00", font=("Courier", 9, "bold"), tags="ws_preview")

    def _is_in_workspace(self, wx, wy):
        """Return True if world point (wx, wy) is inside the defined workspace rect."""
        if self.ws_world_rect is None:
            return True   # no restriction if area not yet defined
        xmin, ymin, xmax, ymax = self.ws_world_rect
        return xmin <= wx <= xmax and ymin <= wy <= ymax

    def _arm_auto_track(self):
        """Toggle ARM state for auto-track trigger."""
        if self.auto_track_armed:
            # Disarm
            self.auto_track_armed = False
            self.auto_track_sent  = False
        else:
            # Arm — only allowed if area is defined
            if self.ws_world_rect is not None:
                self.auto_track_armed = True
        self._update_workspace_ui()
        self.draw_robot()

    def _clear_workspace(self):
        """Reset workspace area to undefined state."""
        self.ws_mode          = "idle"
        self.ws_p1            = None
        self.ws_p2            = None
        self.ws_world_rect    = None
        self.auto_track_armed = False
        self.auto_track_sent  = False
        self.canvas_top.delete("ws_rect")
        self.canvas_top.delete("ws_preview")
        self._update_workspace_ui()
        self.draw_robot()

    def _update_workspace_ui(self):
        """Update workspace panel status label and ARM button state."""
        if self.ws_mode == "idle":
            self.lbl_ws_status.config(
                text="\u25a1 Area: Not defined — switch to 'Define Workspace Area' mode then click 2 pts in Top View",
                foreground="#888888")
            self.btn_arm.config(state='disabled', text="ARM AUTO-TRACK")
        elif self.ws_mode == "drawing":
            self.lbl_ws_status.config(
                text="\u25a3 Drawing... click 2nd point to confirm",
                foreground="#FF8C00")
            self.btn_arm.config(state='disabled', text="ARM AUTO-TRACK")
        elif self.ws_mode == "defined":
            xmin, ymin, xmax, ymax = self.ws_world_rect
            area_txt = f"X:[{xmin:.2f}\u2192{xmax:.2f}]m  Y:[{ymin:.2f}\u2192{ymax:.2f}]m"
            if self.auto_track_sent:
                self.lbl_ws_status.config(
                    text=f"\u25c9 RUNNING...\n{area_txt}",
                    foreground="#00aaff")
                self.btn_arm.config(state='disabled', text="\u23f3 Robot Running...")
            elif self.auto_track_armed:
                self.lbl_ws_status.config(
                    text=f"\u25cf ARMED — waiting for detection\n{area_txt}",
                    foreground="#00FF44")
                self.btn_arm.config(state='normal', text="\u2713 ARMED — Click to Disarm")
            else:
                self.lbl_ws_status.config(
                    text=f"\u25cb Area defined — press ARM to activate\n{area_txt}",
                    foreground="#FF8C00")
                self.btn_arm.config(state='normal', text="ARM AUTO-TRACK")

    def generate_workspace(self):
        self.canvas_top.delete("workspace")
        self.canvas_bottom.delete("workspace")
        for _ in range(1500):
            jnts = [random.uniform(-math.pi, math.pi) for _ in range(self.dof)]
            points, _ = self.calculate_fk(jnts)
            ee = points[-1]
            tx = self.origin_x + ee[0]*self.scale_factor
            ty = self.origin_y - ee[1]*self.scale_factor
            fx = self.origin_x + ee[0]*self.scale_factor
            fz = self.front_origin_y - ee[2]*self.scale_factor
            self.canvas_top.create_oval(tx-1, ty-1, tx+1, ty+1, fill="#4A4A4A", outline="", tags="workspace")
            self.canvas_bottom.create_oval(fx-1, fz-1, fx+1, fz+1, fill="#4A4A4A", outline="", tags="workspace")

    def calculate_fk(self, joints):
        points = [np.array([0, 0, 0])]
        Ts = [np.eye(4)]
        T = np.eye(4)
        for i in range(self.dof):
            a, alpha_deg, d, theta_off_deg = self.dh_params[i]
            alpha = math.radians(alpha_deg)
            theta_off = math.radians(theta_off_deg)
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
            points.append(T[:3, 3])
            Ts.append(T)
            
        return points, Ts

    def render_pip_camera(self, norm_x, norm_y, area):
        self.pip_canvas.delete("all")
        if norm_x >= 0 and not self.is_grabbed:
            u = norm_x * 320.0
            v = norm_y * 240.0
            size = math.sqrt(area * 320.0 * 240.0)
            self.pip_canvas.create_rectangle(u-size/2, v-size/2, u+size/2, v+size/2, outline="#00FF00", width=3)
            self.lbl_telemetry.config(text=f"X: {norm_x:.2f} | Y: {norm_y:.2f} | Area: {area:.3f}")
        else:
            self.lbl_telemetry.config(text="X: --- | Y: --- | Area: ---")

    def draw_robot(self):
        self.canvas_top.delete("dynamic")
        self.canvas_bottom.delete("dynamic")

        # ── Workspace area rectangle (persistent overlay on top view) ──
        self.canvas_top.delete("ws_rect")
        if self.ws_mode == "defined" and self.ws_world_rect:
            xmin, ymin, xmax, ymax = self.ws_world_rect
            px1 = self.origin_x + xmin * self.scale_factor
            py1 = self.origin_y - ymin * self.scale_factor
            px2 = self.origin_x + xmax * self.scale_factor
            py2 = self.origin_y - ymax * self.scale_factor
            if self.auto_track_sent:
                ws_clr = "#00aaff"   # cyan — robot running
            elif self.auto_track_armed:
                ws_clr = "#00FF44"   # green — armed
            else:
                ws_clr = "#FF8C00"   # orange — defined, not armed
            self.canvas_top.create_rectangle(px1, py1, px2, py2,
                outline=ws_clr, width=2, dash=(6, 3), tags="ws_rect")
            lbl = "ARMED" if self.auto_track_armed else ("RUNNING" if self.auto_track_sent else "WORKSPACE")
            self.canvas_top.create_text(
                (px1 + px2) / 2, min(py1, py2) - 10,
                text=lbl, fill=ws_clr,
                font=("Courier", 9, "bold"), tags="ws_rect")
        
        # Legends
        self.canvas_top.create_text(10, 10, text="■ X-Axis (Red)", fill="red", anchor="nw", font=("Helvetica", 10, "bold"), tags="dynamic")
        self.canvas_top.create_text(10, 25, text="■ Y-Axis (Green)", fill="green", anchor="nw", font=("Helvetica", 10, "bold"), tags="dynamic")
        
        self.canvas_bottom.create_text(10, 10, text="■ X-Axis (Red)", fill="red", anchor="nw", font=("Helvetica", 10, "bold"), tags="dynamic")
        self.canvas_bottom.create_text(10, 25, text="■ Z-Axis (Blue)", fill="blue", anchor="nw", font=("Helvetica", 10, "bold"), tags="dynamic")
        
        points, Ts = self.calculate_fk(self.current_joints)
        
        top_ox = self.origin_x
        top_oy = self.origin_y
        front_ox = self.origin_x
        front_oy = self.front_origin_y
        
        # Draw Target Ghost / Detected Object Dot
        if self.cam_mode == "realcam" and not self.is_homing:
            # In real-camera mode: dot is ALWAYS drawn once we have a target.
            # ● Delivery (is_grabbed): follows gripper → magenta
            # ● Detection active (cam_detecting): last ray-cast position → bright cyan
            # ● Detection lost (stale): last known position → dim cyan (dashed)
            # ● is_homing = True: dot hidden (arm traveling home)
            if self.is_grabbed:
                # Gripper carries the object: dot follows EE in ALL axes
                ee = points[-1]
                dot_x, dot_y, dot_z = ee[0], ee[1], ee[2]
                dot_color = "#ff00ff"
                fill_clr  = "#ff00ff"
                r, dash = 8, ()
            elif self.has_target:
                dot_x, dot_y = self.target_x, self.target_y
                dot_z = 0.0   # detected object on floor
                if self.cam_detecting:
                    dot_color = "#00ffcc"   # bright cyan — live
                    fill_clr  = "#00ffcc"
                    r, dash = 8, ()
                else:
                    dot_color = "#009977"   # dim cyan — last known
                    fill_clr  = ""
                    r, dash = 7, (5, 3)
            else:
                dot_x = dot_y = dot_z = None  # no target yet


            if dot_x is not None:
                # ── Top view (XY) ──────────────────────────────────
                tx_px = top_ox + dot_x * self.scale_factor
                ty_px = top_oy - dot_y * self.scale_factor
                self.canvas_top.create_oval(tx_px-r, ty_px-r, tx_px+r, ty_px+r,
                                            fill=fill_clr, outline=dot_color,
                                            width=2, dash=dash, tags="dynamic")
                lbl_top = "gripper" if self.is_grabbed else f"({dot_x:.2f}, {dot_y:.2f})"
                self.canvas_top.create_text(
                    tx_px + r + 4, ty_px,
                    text=lbl_top, fill=dot_color,
                    font=("Courier", 8), anchor="w", tags="dynamic")

                # ── Front view (XZ) ─────────────────────────────────
                fx_px = front_ox + dot_x * self.scale_factor
                fz_px = front_oy - dot_z * self.scale_factor   # correct Z height
                self.canvas_bottom.create_oval(fx_px-r, fz_px-r, fx_px+r, fz_px+r,
                                               fill=fill_clr, outline=dot_color,
                                               width=2, dash=dash, tags="dynamic")
                lbl_front = f"Z={dot_z:.2f}" if self.is_grabbed else "Z=0"
                self.canvas_bottom.create_text(
                    fx_px + r + 4, fz_px,
                    text=lbl_front, fill=dot_color,
                    font=("Courier", 8), anchor="w", tags="dynamic")



        elif self.has_target:  # manual mode
            dot_color = "#ffff00"
            fill_clr  = "" if not self.is_grabbed else "#ff00ff"
            outline   = dot_color if not self.is_grabbed else "#ff00ff"
            r = 5
            tx_px = top_ox + self.target_x * self.scale_factor
            ty_px = top_oy - self.target_y * self.scale_factor
            self.canvas_top.create_oval(tx_px-r, ty_px-r, tx_px+r, ty_px+r,
                                        fill=fill_clr, outline=outline,
                                        width=2, tags="dynamic")
            fx_px = front_ox + self.target_x * self.scale_factor
            fz_px = front_oy - self.target_z * self.scale_factor
            self.canvas_bottom.create_oval(fx_px-r, fz_px-r, fx_px+r, fz_px+r,
                                           fill=fill_clr, outline=outline,
                                           width=2, tags="dynamic")

        # Draw 3D-like Perspective Container
        s_base = 15 # Base half-size 
        dist_y = self.container_pos[1] # Distance from robot origin depth
        
        # Scaling perspective: As Y increases (+Y, moving away), object shrinks. Y=0 is robot base. Viewer is at Y=-1.5.
        p_scale = max(0.3, 1.5 / (1.5 + dist_y))
        s = s_base * p_scale
        
        # Top view (Orthographic square)
        cx_t = top_ox + self.container_pos[0] * self.scale_factor
        cy_t = top_oy - self.container_pos[1] * self.scale_factor
        self.canvas_top.create_rectangle(cx_t-s_base, cy_t-s_base, cx_t+s_base, cy_t+s_base, outline="#00aaff", width=3, dash=(4,2), tags="dynamic")
        
        # Front view (Perspective 3D Box)
        # Floor is at cz_f. Container goes UP from floor.
        cx_f = front_ox + self.container_pos[0] * self.scale_factor
        cz_f = front_oy - self.container_pos[2] * self.scale_factor
        h = s * 0.4 # Height of box (Shortened to a quarter / tray)
        w = s
        
        # Front face
        self.canvas_bottom.create_rectangle(cx_f-w, cz_f-h, cx_f+w, cz_f, outline="#00aaff", width=3, dash=(4,2), tags="dynamic")
        # 3D lines
        offset = w * 0.4
        self.canvas_bottom.create_line(cx_f-w, cz_f-h, cx_f-w+offset, cz_f-h-offset, fill="#00aaff", width=2, dash=(2,2), tags="dynamic")
        self.canvas_bottom.create_line(cx_f+w, cz_f-h, cx_f+w+offset, cz_f-h-offset, fill="#00aaff", width=2, dash=(2,2), tags="dynamic")
        self.canvas_bottom.create_line(cx_f-w+offset, cz_f-h-offset, cx_f+w+offset, cz_f-h-offset, fill="#00aaff", width=2, dash=(2,2), tags="dynamic")
        self.canvas_bottom.create_line(cx_f+w, cz_f, cx_f+w+offset, cz_f-offset, fill="#00aaff", width=2, dash=(2,2), tags="dynamic")
        self.canvas_bottom.create_line(cx_f+w+offset, cz_f-h-offset, cx_f+w+offset, cz_f-offset, fill="#00aaff", width=2, dash=(2,2), tags="dynamic")
        
        # Draw Links
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            
            x1_t, y1_t = top_ox + p1[0]*self.scale_factor, top_oy - p1[1]*self.scale_factor
            x2_t, y2_t = top_ox + p2[0]*self.scale_factor, top_oy - p2[1]*self.scale_factor
            self.canvas_top.create_line(x1_t, y1_t, x2_t, y2_t, fill="#00ffcc", width=4, tags="dynamic")
            self.canvas_top.create_oval(x1_t-4, y1_t-4, x1_t+4, y1_t+4, fill="#ff3366", tags="dynamic")
            
            x1_f, z1_f = front_ox + p1[0]*self.scale_factor, front_oy - p1[2]*self.scale_factor
            x2_f, z2_f = front_ox + p2[0]*self.scale_factor, front_oy - p2[2]*self.scale_factor
            self.canvas_bottom.create_line(x1_f, z1_f, x2_f, z2_f, fill="#00ffcc", width=4, tags="dynamic")
            self.canvas_bottom.create_oval(x1_f-4, z1_f-4, x1_f+4, z1_f+4, fill="#ff3366", tags="dynamic")
            
            if i == len(points)-2: # Last link → replace with Gripper
                rot = Ts[-1][:3, :3]
                x_ax, y_ax, z_ax = rot[:, 0], rot[:, 1], rot[:, 2]
                
                jaw_gap   = 10 if self.gripper_open else 4
                jaw_len   = 14
                gripper_color = "#00ff88" if self.gripper_open else "#ff8800"
                
                app_ax_t = np.array([ x_ax[0], -x_ax[1] ])
                lat_ax_t = np.array([ y_ax[0], -y_ax[1] ])
                app_ax_f = np.array([ x_ax[0], -x_ax[2] ])
                lat_ax_f = np.array([ y_ax[0], -y_ax[2] ])
                
                # TOP: Palm stem from p1→p2 (replaces cyan link), then crossbar + fingers at TCP
                self.canvas_top.create_line(x1_t, y1_t, x2_t, y2_t, fill=gripper_color, width=4, tags="dynamic")
                f1xt = x2_t + lat_ax_t[0]*jaw_gap;  f1yt = y2_t + lat_ax_t[1]*jaw_gap
                f2xt = x2_t - lat_ax_t[0]*jaw_gap;  f2yt = y2_t - lat_ax_t[1]*jaw_gap
                self.canvas_top.create_line(f1xt, f1yt, f2xt, f2yt, fill=gripper_color, width=3, tags="dynamic")
                self.canvas_top.create_line(f1xt, f1yt, f1xt+app_ax_t[0]*jaw_len, f1yt+app_ax_t[1]*jaw_len, fill=gripper_color, width=3, tags="dynamic")
                self.canvas_top.create_line(f2xt, f2yt, f2xt+app_ax_t[0]*jaw_len, f2yt+app_ax_t[1]*jaw_len, fill=gripper_color, width=3, tags="dynamic")
                
                # FRONT: same
                self.canvas_bottom.create_line(x1_f, z1_f, x2_f, z2_f, fill=gripper_color, width=4, tags="dynamic")
                f1xf = x2_f + lat_ax_f[0]*jaw_gap;  f1zf = z2_f + lat_ax_f[1]*jaw_gap
                f2xf = x2_f - lat_ax_f[0]*jaw_gap;  f2zf = z2_f - lat_ax_f[1]*jaw_gap
                self.canvas_bottom.create_line(f1xf, f1zf, f2xf, f2zf, fill=gripper_color, width=3, tags="dynamic")
                self.canvas_bottom.create_line(f1xf, f1zf, f1xf+app_ax_f[0]*jaw_len, f1zf+app_ax_f[1]*jaw_len, fill=gripper_color, width=3, tags="dynamic")
                self.canvas_bottom.create_line(f2xf, f2zf, f2xf+app_ax_f[0]*jaw_len, f2zf+app_ax_f[1]*jaw_len, fill=gripper_color, width=3, tags="dynamic")
                
                # TCP dot at jaw root (where object is held)
                self.canvas_top.create_oval(x2_t-4, y2_t-4, x2_t+4, y2_t+4, fill=gripper_color, tags="dynamic")
                self.canvas_bottom.create_oval(x2_f-4, z2_f-4, x2_f+4, z2_f+4, fill=gripper_color, tags="dynamic")
                
                # RGB Axes (smaller, at TCP)
                ax_len = 20
                self.canvas_top.create_line(x2_t, y2_t, x2_t + x_ax[0]*ax_len, y2_t - x_ax[1]*ax_len, fill="red", width=2, tags="dynamic")
                self.canvas_top.create_line(x2_t, y2_t, x2_t + y_ax[0]*ax_len, y2_t - y_ax[1]*ax_len, fill="green", width=2, tags="dynamic")
                self.canvas_top.create_line(x2_t, y2_t, x2_t + z_ax[0]*ax_len, y2_t - z_ax[1]*ax_len, fill="blue", width=2, tags="dynamic")
                self.canvas_bottom.create_line(x2_f, z2_f, x2_f + x_ax[0]*ax_len, z2_f - x_ax[2]*ax_len, fill="red", width=2, tags="dynamic")
                self.canvas_bottom.create_line(x2_f, z2_f, x2_f + y_ax[0]*ax_len, z2_f - y_ax[2]*ax_len, fill="green", width=2, tags="dynamic")
                self.canvas_bottom.create_line(x2_f, z2_f, x2_f + z_ax[0]*ax_len, z2_f - z_ax[2]*ax_len, fill="blue", width=2, tags="dynamic")

                # ── Draw Camera FOV (Pyramid Projection) ──
                if not self.is_homing:
                    fov_color = "#ffaa00"  # Orange/amber color for FOV
                    T_ee = Ts[-1]
                    P_tcp = points[-1]
                    fov_base_t = []
                    fov_base_f = []
                    
                    # TL, TR, BR, BL corners of a 320x240 image with virtual fx=250
                    for cy_r, cx_r in [(-0.48, -0.64), (-0.48, 0.64), (0.48, 0.64), (0.48, -0.64)]:
                        ray_cam = np.array([1.0, cy_r, -cx_r, 0.0])
                        ray_world = T_ee @ ray_cam
                        X_vec = ray_world[:3]
                        
                        # Ray length: exactly intersect the Z=0 floor, OR cap at 1.0m if looking up
                        if abs(X_vec[2]) > 1e-3 and (-P_tcp[2] / X_vec[2]) > 0:
                            lam = min(-P_tcp[2] / X_vec[2], 1.2)
                        else:
                            lam = 1.2
                            
                        hit = P_tcp + lam * X_vec
                        
                        xt = top_ox + hit[0]*self.scale_factor
                        yt = top_oy - hit[1]*self.scale_factor
                        xf = front_ox + hit[0]*self.scale_factor
                        zf = front_oy - hit[2]*self.scale_factor
                        
                        fov_base_t.extend([xt, yt])
                        fov_base_f.extend([xf, zf])
                        
                    # Draw the projected footprint polygon
                    self.canvas_top.create_polygon(fov_base_t, outline=fov_color, fill="", dash=(2,4), tags="dynamic")
                    self.canvas_bottom.create_polygon(fov_base_f, outline=fov_color, fill="", dash=(2,4), tags="dynamic")
                    
                    # Draw the 4 pyramid edges originating from the camera (TCP)
                    for idx in range(4):
                        self.canvas_top.create_line(x2_t, y2_t, fov_base_t[idx*2], fov_base_t[idx*2+1], fill=fov_color, dash=(2,4), tags="dynamic")
                        self.canvas_bottom.create_line(x2_f, z2_f, fov_base_f[idx*2], fov_base_f[idx*2+1], fill=fov_color, dash=(2,4), tags="dynamic")

        # Base Base RGB Axes
        ax_len = 40
        self.canvas_top.create_line(top_ox, top_oy, top_ox + ax_len, top_oy, fill="red", width=2, tags="dynamic")
        self.canvas_top.create_line(top_ox, top_oy, top_ox, top_oy - ax_len, fill="green", width=2, tags="dynamic")
        self.canvas_bottom.create_line(front_ox, front_oy, front_ox + ax_len, front_oy, fill="red", width=2, tags="dynamic")
        self.canvas_bottom.create_line(front_ox, front_oy, front_ox, front_oy - ax_len, fill="blue", width=2, tags="dynamic")

    def update_loop(self):
        has_new = False
        try:
            while True:
                jnts = self.node.msg_queue.get_nowait()
                if len(jnts) == self.dof:
                    self.current_joints = jnts
                    has_new = True
        except queue.Empty: pass
            
        try:
            while True:
                tx, ty, tz = self.node.target_queue.get_nowait()
                if tx == -9.0 and ty == -9.0 and tz == -9.0:
                    # Sentinel: drop complete — unlock so ray-cast resumes
                    self.cam_target_locked = False
                    continue
                # Authoritative position from controller — lock in
                self.target_x, self.target_y, self.target_z = tx, ty, tz
                self.has_target        = True
                self.cam_target_locked = True   # stop raw ray-cast from overwriting
                self.var_x.set(f"{tx:.3f}")
                self.var_y.set(f"{ty:.3f}")
                self.var_z.set(f"{tz:.3f}")
                has_new = True
        except queue.Empty: pass
            

            
        try:
            if hasattr(self.node, 'cmds_rcvd'):
                while True:
                    cmd_str = self.node.cmds_rcvd.get_nowait()
                    if cmd_str == "ATTACH_OBJECT":
                        self.is_grabbed = True
                        self.gripper_open = False
                    elif cmd_str == "DETACH_OBJECT":
                        self.is_grabbed = False
                        self.gripper_open = True
        except queue.Empty: pass

        # ── Robot status (HOMING / IDLE / SCANNING / …) ───────────
        if hasattr(self.node, 'status_queue'):
            try:
                while True:
                    status = self.node.status_queue.get_nowait()
                    if status == "HOMING":
                        # Arm traveling back to home — hide dot
                        self.is_homing         = True
                        self.cam_target_locked = False  # unlock so dot resumes at home
                        self.cam_detecting     = False
                    elif status == "IDLE":
                        # Arm arrived at home — restore visualization
                        self.is_homing       = False
                        self.auto_track_sent = False   # cycle complete, reset flag
                        # Re-arm logic based on selected mode
                        if (self.rearm_mode.get() == "autoloop"
                                and self.ws_world_rect is not None
                                and self.ws_mode == "defined"):
                            self.auto_track_armed = True
                        self._update_workspace_ui()
                    has_new = True
            except queue.Empty: pass


        if self.is_grabbed:
            points, _ = self.calculate_fk(self.current_joints)
            self.target_x, self.target_y, self.target_z = points[-1]
            self.var_x.set(f"{self.target_x:.3f}")
            self.var_y.set(f"{self.target_y:.3f}")
            self.var_z.set(f"{self.target_z:.3f}")
            has_new = True
        else:
            # Gravity: visually drop the ungrabbed object to the floor (Z=0.0) 
            if self.target_z > 0.0:
                self.target_z = max(0.0, self.target_z - 0.01)
                self.var_z.set(f"{self.target_z:.3f}")
                has_new = True

        if has_new:
            for i in range(self.dof):
                self.lbl_joints[i].config(text=f"θ{i+1}: {math.degrees(self.current_joints[i]):.2f}°")
                
            points, Ts = self.calculate_fk(self.current_joints)
            ee = points[-1]
            self.lbl_tcp.config(text=f"TCP: X={ee[0]:.3f} Y={ee[1]:.3f} Z={ee[2]:.3f}")
            
            self.draw_robot()
            
        if hasattr(self.node, 'bbox_queue'):
            # ── Drain queue (inner try so post-loop code ALWAYS runs) ──
            last_bx = last_by = last_barea = None
            try:
                while True:
                    bx, by, barea = self.node.bbox_queue.get_nowait()
                    self.render_pip_camera(bx, by, barea)
                    last_bx, last_by, last_barea = bx, by, barea
            except queue.Empty:
                pass   # expected — queue is now empty

            # ── Post-drain: update target from last bbox ──────────────
            # Skip when controller has already locked an authoritative position
            if last_bx is not None and self.cam_mode == "realcam" \
                    and not self.is_grabbed and not self.cam_target_locked:
                if last_bx >= 0:
                    result = self.ray_cast_to_floor(last_bx, last_by)
                    if result is not None:
                        wx, wy = result
                        max_reach = sum(abs(row[0]) for row in self.dh_params)
                        dist = math.sqrt(wx*wx + wy*wy)
                        if dist > max_reach:
                            wx *= max_reach / dist
                            wy *= max_reach / dist
                        self.target_x      = wx
                        self.target_y      = wy
                        self.target_z      = 0.0
                        self.has_target    = True
                        self.cam_detecting = True
                        self.var_x.set(f"{wx:.3f}")
                        self.var_y.set(f"{wy:.3f}")
                        self.var_z.set("0.000")
                        # ── Auto-track trigger: fire START if ARMED + in workspace ──
                        if (self.auto_track_armed
                                and not self.auto_track_sent
                                and self._is_in_workspace(wx, wy)):
                            self.node.publish_command("START")
                            self.auto_track_armed = False
                            self.auto_track_sent  = True
                            self._update_workspace_ui()
                else:
                    # No detection — keep last known position, mark stale
                    self.cam_detecting = False

            # Redraw canvas on EVERY bbox cycle (even idle/home robot)
            if last_bx is not None and self.cam_mode == "realcam":
                self.draw_robot()





        self.root.after(50, self.update_loop)



def main():
    if ROS2_AVAILABLE:
        rclpy.init()
    
    node = VizEnvNode()
    
    if ROS2_AVAILABLE:
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()

    root = tk.Tk()
    app = RobotGUI(root, node)
    root.mainloop()

    if ROS2_AVAILABLE:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
