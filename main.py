"""
mecanum_mpc_rerun_rotated.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Solve the smooth-MPC problem for a mecanum robot (or differential drive),
then visualise the rollout in Rerun.

pip install casadi numpy rerun-sdk
"""

import numpy as np
import casadi as ca
import argparse
import utils         # Import helper functions
import visualization # Import Rerun logging functions
from utils import Obstacle # Import the Obstacle dataclass


# ------------------------------------------------------------
# 0.  Problem constants & Setup
# ------------------------------------------------------------
r, lx, ly = 0.05, 0.20, 0.15            # wheel radius & chassis half-sizes (m)
dt, N = 0.1, 500                         # sample time & horizon (s) - Reduced N for quicker testing
v_max, v_min = 20.0, -20.0              # wheel-speed limits (m/s)
a_max = 5.0                            # wheel accel limits (m/s)
goal_pose = ca.DM([1.8, 1.2, np.pi/2])  # x, y, θ target (m, m, rad)

# Obstacles: Instantiate Obstacle dataclasses
obstacles = [
    Obstacle(x=0.80, y=0.50, width=0.40, height=0.30, yaw=0.40),    # 23° turned
    Obstacle(x=1.10, y=1.20, width=0.30, height=0.30, yaw=-0.30),   # −17°
    Obstacle(x=1.50, y=0.50, width=0.40, height=0.30, yaw=0.40)
]

# Calculate bounding radii and robot shape
robot_width, robot_length = 0.40, 0.30
# obs_R = [0.5 * np.hypot(w, h) for _, _, w, h, _ in obstacles] # No longer needed
R_robot = 0.5 * np.hypot(robot_width, robot_length)
robot_half_size = np.array([robot_width / 2, robot_length / 2], np.float32)

# ------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description='Run MPC for mecanum or differential drive robot.')
parser.add_argument('--differential-drive', action='store_true',
                    help='Use differential drive (tank) kinematics instead of mecanum.')
args = parser.parse_args()

# Select kinematics function based on args
if args.differential_drive:
    body_twist_func = lambda w: utils.body_twist_tank(w, r, ly)
    print("Using Differential Drive (Tank) Kinematics")
else:
    body_twist_func = lambda w: utils.body_twist_mecanum(w, r, lx, ly)
    print("Using Mecanum Drive Kinematics")

# ------------------------------------------------------------
# Perform Initial Goal Validation
# ------------------------------------------------------------
# Pass the list of Obstacle objects directly
if not utils.is_valid_goal(goal_pose, obstacles, R_robot):
    raise ValueError("Initial goal pose is invalid due to collision with an obstacle.")
else:
    print("Initial goal pose is valid.")

# ------------------------------------------------------------
# 1.  Build smooth MPC
# ------------------------------------------------------------
opti = ca.Opti()
X = opti.variable(7, N + 1)  # [x,y,θ, ω1..ω4]
U = opti.variable(4, N)  # wheel accelerations
x0 = opti.parameter(7)
opti.subject_to(X[:, 0] == x0)

for k in range(N):
    # --- Extract current state ---
    p_k = X[0:3, k]  # Pose [x, y, th]
    w = X[3:7, k]  # Wheel speeds
    a = U[:, k]  # Wheel accelerations

    # --- Calculate Body Twist (using selected function) ---
    vx_b, vy_b, vth = body_twist_func(w)
    xi_k = ca.vcat([vx_b, vy_b, vth])

    # --- SE(2) Integration ---
    P_k = utils.pose_to_matrix(p_k)
    T_dt = utils.exp_se2(xi_k, dt)
    P_k_plus_1 = ca.mtimes(P_k, T_dt)
    p_k_plus_1 = utils.matrix_to_pose(P_k_plus_1)

    # --- Set Dynamics Constraints ---
    opti.subject_to(X[0:3, k + 1] == p_k_plus_1)  # Pose update
    opti.subject_to(X[3:7, k + 1] == w + dt * a)  # Wheel dynamics (Euler)

    # --- State & Control Constraints ---
    opti.subject_to(opti.bounded(v_min, X[3:7, k], v_max))
    opti.subject_to(opti.bounded(-a_max, a, a_max))

    # --- Obstacle Avoidance ---
    px, py = p_k_plus_1[0], p_k_plus_1[1]
    # Iterate through Obstacle objects
    for obstacle in obstacles:
        # Use attributes from the Obstacle dataclass
        d = ca.sqrt((px - obstacle.x)**2 + (py - obstacle.y)**2) - (R_robot + obstacle.radius)
        opti.subject_to(d >= 5e-2)
        opti.minimize(-0.02 * ca.log(d))

# --- Cost Function ---
Qp, Qw, R = np.diag([25, 25, 35]), 0.15 * np.eye(4), 0.01 * np.eye(4)
obj = 0
for k in range(N):
    e = X[0:3, k] - goal_pose
    obj += ca.mtimes([e.T, Qp, e])
    obj += ca.mtimes([X[3:7, k].T, Qw, X[3:7, k]])
    obj += ca.mtimes([U[:, k].T, R, U[:, k]])
eT = X[0:3, N] - goal_pose
obj += ca.mtimes([eT.T, 100 * Qp, eT])
wT = X[3:7, N]
obj += ca.mtimes([wT.T, 3 * Qw, wT])
opti.minimize(obj)
opti.solver("ipopt", {"print_time": False}, {"max_iter": 400})

# ------------------------------------------------------------
# 2.  Solve MPC
# ------------------------------------------------------------
state0 = np.zeros(7)
opti.set_value(x0, state0)
sol = opti.solve()
X_val = np.array(sol.value(X))  # (7, N+1)

# ------------------------------------------------------------
# 3.  Rerun Visualisation
# ------------------------------------------------------------
visualization.init_rerun()
visualization.log_static_scene(
    goal_pose=goal_pose,
    obstacles=obstacles, # Pass the list of Obstacle objects
    # obs_R=obs_R, # No longer needed
    robot_half_size=robot_half_size
)

path_pts = []
for k in range(N + 1):
    # Extract state at step k
    x, y, th, w1, w2, w3, w4 = X_val[:, k]

    visualization.log_timestep(
        k=k,
        x=x, y=y, th=th,
        w1=w1, w2=w2, w3=w3, w4=w4,
        path_pts=path_pts, # Pass list to append to
        R_robot=R_robot
    )

print("✨ Simulation complete. Check Rerun viewer.")

