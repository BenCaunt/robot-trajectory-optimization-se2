"""
mecanum_mpc_rerun_rotated.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Solve the smooth-MPC problem for a mecanum robot, then visualise the rollout
in Rerun with **properly oriented** rectangular robot + obstacles.

pip install casadi numpy rerun-sdk
"""

import numpy as np
import casadi as ca
import rerun as rr


# ------------------------------------------------------------
# 0.  Problem constants
# ------------------------------------------------------------
r, lx, ly = 0.05, 0.20, 0.15            # wheel radius & chassis half-sizes (m)
dt, N = 0.1, 500                         # sample time & horizon (s)
v_max, v_min = 20.0, -20.0              # wheel-speed limits (m/s)
a_max = 5.0                            # wheel accel limits (m/s)
goal_pose = ca.DM([1.8, 1.2, np.pi/2])  # x, y, θ target (m, m, rad)

# ------------------------------------------------------------
# SE(2) Helper Functions (CasADi)
# ------------------------------------------------------------

def pose_to_matrix(p):
    """Convert [x, y, th] pose vector to 3x3 SE(2) matrix."""
    x, y, th = p[0], p[1], p[2]
    cos_th = ca.cos(th)
    sin_th = ca.sin(th)
    return ca.vcat([
        ca.hcat([cos_th, -sin_th, x]),
        ca.hcat([sin_th,  cos_th, y]),
        ca.hcat([0,       0,      1])
    ])

def matrix_to_pose(M):
    """Convert 3x3 SE(2) matrix back to [x, y, th] pose vector."""
    x = M[0, 2]
    y = M[1, 2]
    th = ca.atan2(M[1, 0], M[0, 0]) # Use atan2 for robustness
    return ca.vcat([x, y, th])

def sinc(x):
    """ Numerically stable sinc function sin(x)/x. """
    # Use Taylor expansion near x=0
    return ca.if_else(ca.fabs(x) < 1e-9, 1 - x**2 / 6, ca.sin(x) / x)

def exp_se2(twist, dt):
    """Computes the SE(2) exponential map exp(hat(twist * dt)).

    Args:
        twist: CasADi symbolic vector [vx, vy, vth].
        dt: Time step.

    Returns:
        3x3 SE(2) matrix.
    """
    vx, vy, vth = twist[0], twist[1], twist[2]
    phi = vth * dt

    # Translational component (V matrix)
    V_11 = sinc(phi / 2) * ca.cos(phi / 2)
    V_12 = -sinc(phi / 2) * ca.sin(phi / 2)
    V_21 = sinc(phi / 2) * ca.sin(phi / 2)
    V_22 = sinc(phi / 2) * ca.cos(phi / 2)
    V = ca.vcat([
        ca.hcat([V_11, V_12]),
        ca.hcat([V_21, V_22])
    ])

    # Rotational component (rotation matrix)
    cos_phi = ca.cos(phi)
    sin_phi = ca.sin(phi)
    R = ca.vcat([
        ca.hcat([cos_phi, -sin_phi]),
        ca.hcat([sin_phi,  cos_phi])
    ])

    # Translation vector
    trans = ca.mtimes(V, ca.vcat([vx, vy])) * dt

    # Combine into SE(2) matrix using hcat/vcat
    top_block = ca.hcat([R, trans])         # Shape (2, 3)
    bottom_row = ca.MX(ca.DM([[0, 0, 1]])) # Force MX type to match top_block
    T = ca.vcat([top_block, bottom_row])    # Shape (3, 3)
    return T

# ------------------------------------------------------------
# Rerun Helper Functions
# ------------------------------------------------------------
def generate_circle_points(center_x, center_y, radius, num_points=50):
    """Generates points for a circle."""
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    # Need to close the loop for LineStrips2D
    points = np.vstack([x, y]).T
    return np.vstack([points, points[0]]) # Append first point to the end


# Obstacles: (x, y, w, h, yaw)  – yaw in *radians*
obstacles = [
    (0.80, 0.50, 0.40, 0.30, 0.40),    # 23° turned
    (1.10, 1.20, 0.30, 0.30, -0.30),   # −17°
    (1.50, 0.5, 0.4, 0.3, 0.4)
]

obs_R  = [0.5*np.hypot(w, h) for _,_,w,h,_ in obstacles]
R_robot = 0.5*np.hypot(0.40, 0.30)

# ------------------------------------------------------------
# Validation Function
# ------------------------------------------------------------
def is_valid_goal(goal_pose, obstacles, R_robot, obs_R):
    """Checks if the goal pose is collision-free w.r.t obstacle bounding circles."""
    goal_x, goal_y = float(goal_pose[0]), float(goal_pose[1])
    for i, (ox, oy, *_) in enumerate(obstacles):
        dist_sq = (goal_x - ox)**2 + (goal_y - oy)**2
        min_dist_sq = (R_robot + obs_R[i])**2
        if dist_sq < min_dist_sq:
            print(f"Collision detected between goal and obstacle {i}!")
            print(f"  Goal: ({goal_x:.2f}, {goal_y:.2f}), Robot Radius: {R_robot:.2f}")
            print(f"  Obstacle {i}: ({ox:.2f}, {oy:.2f}), Obstacle Radius: {obs_R[i]:.2f}")
            print(f"  Distance: {np.sqrt(dist_sq):.2f}, Required: {np.sqrt(min_dist_sq):.2f}")
            return False
    return True

# ------------------------------------------------------------
# Perform Initial Goal Validation
# ------------------------------------------------------------
if not is_valid_goal(goal_pose, obstacles, R_robot, obs_R):
    raise ValueError("Initial goal pose is invalid due to collision with an obstacle.")
else:
    print("Initial goal pose is valid.")

# ------------------------------------------------------------
# 1.  Helper: body twist from wheel speeds
# ------------------------------------------------------------
def body_twist_mecanum(w):
    w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
    vx  = r/4 * ( w1 +  w2 +  w3 +  w4)
    vy  = r/4 * (-w1 +  w2 +  w3 -  w4)
    vth = r/(4*(lx+ly)) * (-w1 + w2 - w3 + w4)
    return vx, vy, vth
    
def body_twist_tank(w):
    # Assumes wheels 1 & 3 are left, 2 & 4 are right
    # Assumes track width is 2*ly
    w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
    wl = (w1 + w3) / 2.0 # Average left speed
    wr = (w2 + w4) / 2.0 # Average right speed

    vx  = r * (wl + wr) / 2.0
    vy  = 0 # No sideways motion for simple tank drive
    vth = r * (wr - wl) / (2 * ly) # Rotation based on speed difference and track width
    return vx, vy, vth


# ------------------------------------------------------------
# 2.  Build smooth MPC (unchanged)
# ------------------------------------------------------------
opti = ca.Opti()
X  = opti.variable(7, N+1)     # [x,y,θ, ω1..ω4]
U  = opti.variable(4, N)       # wheel accelerations
x0 = opti.parameter(7)
opti.subject_to(X[:,0] == x0)

for k in range(N):
    # --- Extract current state --- 
    p_k = X[0:3,k] # Pose [x, y, th]
    w   = X[3:7,k] # Wheel speeds
    a   = U[:,k]   # Wheel accelerations

    # --- Calculate Body Twist --- 
    vx_b, vy_b, vth = body_twist_tank(w)
    xi_k = ca.vcat([vx_b, vy_b, vth])

    # --- SE(2) Integration --- 
    P_k = pose_to_matrix(p_k)
    T_dt = exp_se2(xi_k, dt)
    P_k_plus_1 = ca.mtimes(P_k, T_dt)
    p_k_plus_1 = matrix_to_pose(P_k_plus_1)

    # --- Set Dynamics Constraints --- 
    opti.subject_to(X[0:3, k+1] == p_k_plus_1) # Pose update
    opti.subject_to(X[3:7,k+1] == w + dt*a)    # Wheel dynamics (Euler)

    # --- State & Control Constraints --- 
    opti.subject_to(opti.bounded(v_min, X[3:7,k], v_max))
    opti.subject_to(opti.bounded(-a_max, a, a_max))

    # --- Obstacle Avoidance (using extracted pose components) --- 
    px, py, th = p_k_plus_1[0], p_k_plus_1[1], p_k_plus_1[2] # Use next pose for constraints
    for (ox,oy, *_), R_o in zip(obstacles, obs_R):
        d = ca.sqrt((px-ox)**2 + (py-oy)**2) - (R_robot + R_o)
        opti.subject_to(d >= 5e-2)
        opti.minimize(-0.02*ca.log(d))
# weight matrices
# Qp: pose error
# Qw: wheel speed error
# R: control input
Qp, Qw, R = np.diag([25,25,35]), 0.15*np.eye(4), 0.01*np.eye(4)
obj = 0
for k in range(N):
    # error of pose
    e = X[0:3,k] - goal_pose
    obj += ca.mtimes([e.T, Qp, e])
    # error of wheel speeds
    obj += ca.mtimes([X[3:7,k].T, Qw, X[3:7,k]])
    # control input
    obj += ca.mtimes([U[:,k].T, R, U[:,k]])
eT = X[0:3,N] - goal_pose
obj += ca.mtimes([eT.T, 100*Qp, eT])
wT = X[3:7,N]
obj += ca.mtimes([wT.T, 3*Qw, wT])
opti.minimize(obj)
opti.solver("ipopt", {"print_time": False}, {"max_iter": 400})

# ------------------------------------------------------------
# 3.  Solve once from rest
# ------------------------------------------------------------
state0 = np.zeros(7)
opti.set_value(x0, state0)
sol = opti.solve()
X_val = np.array(sol.value(X))          # (7, N+1)

# ------------------------------------------------------------
# 4.  Rerun visualisation with yaw-aware boxes
# ------------------------------------------------------------
rr.init("mecanum_mpc_rotated",spawn=True)

# ---- Log goal pose indicator (static) ------------------------------------
goal_x, goal_y, _ = float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2]) # Ignore theta for point
rr.log(
    "world/goal_indicator",
    rr.Points2D(
        positions=[[goal_x, goal_y]],
        colors=[[0, 255, 0, 255]], # Green
        radii=[0.02]               # Small radius
    ),
)

# ---- (4-a) static robot *shape* once (local frame) -------------------------
robot_half = np.array([0.40/2, 0.30/2], np.float32)
rr.log(
    "world/robot",
    rr.Boxes2D(centers=[[0, 0]], half_sizes=[robot_half]),
    static=True,
)

# ---- (4-b) obstacles with static Transform3D + box -------------------------
for i, (ox, oy, w, h, yaw) in enumerate(obstacles):
    path = f"world/obstacles/obs{i}"

    # parent→child transform = translation + yaw (about +Z)
    rr.log(
        path,
        rr.Transform3D(
            translation=[ox, oy, 0.0],
            rotation=rr.RotationAxisAngle((0, 0, 1), radians=float(yaw)),
        ),
        static=True, # Obstacles are static in the world
    )

    # local‐frame, axis-aligned rectangle (logged to a child path)
    rr.log(
        f"{path}/box",
        rr.Boxes2D(
            centers=[[0, 0]],
            half_sizes=[[w / 2, h / 2]],
        ),
        static=True,
    )

    # Log bounding circle (static, also child path)
    circle_pts = generate_circle_points(0, 0, obs_R[i]) # Centered at local origin
    rr.log(
        f"{path}/bounding_circle",
        rr.LineStrips2D([circle_pts.astype(np.float32)], colors=[[255, 255, 0, 128]]), # Yellow, semi-transparent
        static=True, # Bounding circle can also be static
    )

# ---- (4-c) robot pose transform + growing path + wheel velocities --------
path_pts = []
# Iterate through columns (time steps) of X_val
for k in range(N + 1):
    rr.set_time("step", sequence=k)

    # Extract state at step k
    x, y, th, w1, w2, w3, w4 = X_val[:, k]

    # Log robot pose transform
    rr.log(
        "world/robot",
        rr.Transform3D(
            translation=[float(x), float(y), 0.0],
            rotation=rr.RotationAxisAngle((0, 0, 1), radians=float(th)),
        ),
    )

    # Log robot bounding circle (dynamic)
    robot_circle_pts = generate_circle_points(0, 0, R_robot) # Centered at local origin
    rr.log(
        "world/robot/bounding_circle",
        rr.LineStrips2D([robot_circle_pts.astype(np.float32)], colors=[[0, 255, 255, 128]]), # Cyan, semi-transparent
    )

    # Log path points
    path_pts.append([float(x), float(y)])
    rr.log(
        "world/path",
        rr.LineStrips2D([np.array(path_pts, np.float32)]),
    )

    # Log wheel velocities
    rr.log("velocities/w1", rr.Scalars(float(w1)))
    rr.log("velocities/w2", rr.Scalars(float(w2)))
    rr.log("velocities/w3", rr.Scalars(float(w3)))
    rr.log("velocities/w4", rr.Scalars(float(w4)))

print("✨  Obstacles and robot now rotate correctly (Transform3D via rr.log)!")

