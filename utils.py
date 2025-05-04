import numpy as np
import casadi as ca
from dataclasses import dataclass

@dataclass
class Obstacle:
    """Represents a rectangular obstacle with pose and dimensions."""
    x: float
    y: float
    width: float
    height: float
    yaw: float
    radius: float = 0.0 # Will be calculated in __post_init__

    def __post_init__(self):
        """Calculate the bounding radius after initialization."""
        self.radius = 0.5 * np.hypot(self.width, self.height)

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
# Geometry Helper Functions
# ------------------------------------------------------------
def generate_circle_points(center_x, center_y, radius, num_points=50):
    """Generates points for a circle."""
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    # Need to close the loop for LineStrips2D
    points = np.vstack([x, y]).T
    return np.vstack([points, points[0]]) # Append first point to the end

# ------------------------------------------------------------
# Kinematics Helper Functions
# ------------------------------------------------------------
def body_twist_mecanum(w, r, lx, ly):
    """Calculate body twist [vx, vy, vth] from wheel speeds for mecanum drive."""
    w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
    vx  = r/4 * ( w1 +  w2 +  w3 +  w4)
    vy  = r/4 * (-w1 +  w2 +  w3 -  w4)
    vth = r/(4*(lx+ly)) * (-w1 + w2 - w3 + w4)
    return vx, vy, vth

def body_twist_tank(w, r, ly):
    """Calculate body twist [vx, vy, vth] from wheel speeds for tank (differential) drive."""
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
# Validation Function
# ------------------------------------------------------------
def is_valid_goal(goal_pose, obstacles: list[Obstacle], R_robot):
    """Checks if the goal pose is collision-free w.r.t obstacle bounding circles."""
    goal_x, goal_y = float(goal_pose[0]), float(goal_pose[1])
    for i, obstacle in enumerate(obstacles):
        dist_sq = (goal_x - obstacle.x)**2 + (goal_y - obstacle.y)**2
        min_dist_sq = (R_robot + obstacle.radius)**2
        if dist_sq < min_dist_sq:
            print(f"Collision detected between goal and obstacle {i}!")
            print(f"  Goal: ({goal_x:.2f}, {goal_y:.2f}), Robot Radius: {R_robot:.2f}")
            print(f"  Obstacle {i}: ({obstacle.x:.2f}, {obstacle.y:.2f}), Obstacle Radius: {obstacle.radius:.2f}")
            print(f"  Distance: {np.sqrt(dist_sq):.2f}, Required: {np.sqrt(min_dist_sq):.2f}")
            return False
    return True 