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
r, lx, ly = 0.05, 0.20, 0.15            # wheel radius & chassis half-sizes
dt, N = 0.1, 500                         # sample time & horizon
v_max, v_min = 20.0, -20.0              # wheel-speed limits
a_max = 40.0                            # wheel accel limits
goal_pose = ca.DM([1.5, 1.0, np.pi/2])  # x, y, θ target

# Obstacles: (x, y, w, h, yaw)  – yaw in *radians*
obstacles = [
    (0.80, 0.50, 0.40, 0.30, 0.40),    # 23° turned
    (1.10, 1.20, 0.30, 0.30, -0.30),   # −17°
]

obs_R  = [0.5*np.hypot(w, h) for _,_,w,h,_ in obstacles]
R_robot = 0.5*np.hypot(0.40, 0.30)

# ------------------------------------------------------------
# 1.  Helper: body twist from wheel speeds
# ------------------------------------------------------------
def body_twist(w):
    w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
    vx  = r/4 * ( w1 +  w2 +  w3 +  w4)
    vy  = r/4 * (-w1 +  w2 +  w3 -  w4)
    vth = r/(4*(lx+ly)) * (-w1 + w2 - w3 + w4)
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
    px, py, th = X[0,k], X[1,k], X[2,k]
    w          = X[3:7,k]
    a          = U[:,k]

    vx_b, vy_b, vth = body_twist(w)
    v_world = ca.vcat([ca.cos(th)*vx_b - ca.sin(th)*vy_b,
                       ca.sin(th)*vx_b + ca.cos(th)*vy_b])
    opti.subject_to(X[0,k+1] == px + dt*v_world[0])
    opti.subject_to(X[1,k+1] == py + dt*v_world[1])
    opti.subject_to(X[2,k+1] == th + dt*vth)
    opti.subject_to(X[3:7,k+1] == w + dt*a)
    opti.subject_to(opti.bounded(v_min, X[3:7,k], v_max))
    opti.subject_to(opti.bounded(-a_max, a, a_max))

    for (ox,oy, *_), R_o in zip(obstacles, obs_R):
        d = ca.sqrt((px-ox)**2 + (py-oy)**2) - (R_robot + R_o)
        opti.subject_to(d >= 5e-2)
        opti.minimize(-0.02*ca.log(d))
# weight matrices
# Qp: pose error
# Qw: wheel speed error
# R: control input
Qp, Qw, R = np.diag([30,30,10]), 0.2*np.eye(4), 0.01*np.eye(4)
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
obj += ca.mtimes([wT.T, 10*Qw, wT])
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
        timeless=True,
    )

    # local‐frame, axis-aligned rectangle
    rr.log(
        path,
        rr.Boxes2D(
            centers=[[0, 0]],
            half_sizes=[[w / 2, h / 2]],
        ),
        static=True,
    )

# ---- (4-c) robot pose transform + growing path every step ------------------
path_pts = []
for k, (x, y, th) in enumerate(X_val[[0, 1, 2], :].T):
    rr.set_time_sequence("step", k)

    rr.log(
        "world/robot",
        rr.Transform3D(
            translation=[float(x), float(y), 0.0],
            rotation=rr.RotationAxisAngle((0, 0, 1), radians=float(th)),
        ),
    )

    path_pts.append([float(x), float(y)])
    rr.log(
        "world/path",
        rr.LineStrips2D([np.array(path_pts, np.float32)]),
    )

print("✨  Obstacles and robot now rotate correctly (Transform3D via rr.log)!")

