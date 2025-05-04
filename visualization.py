import rerun as rr
import numpy as np
import utils # Assuming utils.py contains generate_circle_points


def init_rerun(name="mecanum_mpc_rotated"):
    """Initialize Rerun."""
    rr.init(name, spawn=True)

def log_static_scene(
    goal_pose,
    obstacles,
    obs_R,
    robot_half_size
):
    """Log static elements like goal, obstacles, and robot shape."""
    # ---- Log goal pose indicator (static) ----
    goal_x, goal_y, _ = float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])
    rr.log(
        "world/goal_indicator",
        rr.Points2D(
            positions=[[goal_x, goal_y]],
            colors=[[0, 255, 0, 255]], # Green
            radii=[0.02]
        ),
        static=True
    )

    # ---- Log static robot shape (local frame) ----
    rr.log(
        "world/robot",
        rr.Boxes2D(centers=[[0, 0]], half_sizes=[robot_half_size]),
        static=True,
    )

    # ---- Log obstacles with static Transform3D + box + circle ----
    for i, (ox, oy, w, h, yaw) in enumerate(obstacles):
        path = f"world/obstacles/obs{i}"

        # Log transform (static)
        rr.log(
            path,
            rr.Transform3D(
                translation=[ox, oy, 0.0],
                rotation=rr.RotationAxisAngle((0, 0, 1), radians=float(yaw)),
            ),
            static=True,
        )

        # Log box geometry (static, child path)
        rr.log(
            f"{path}/box",
            rr.Boxes2D(
                centers=[[0, 0]],
                half_sizes=[[w / 2, h / 2]],
            ),
            static=True,
        )

        # Log bounding circle (static, child path)
        circle_pts = utils.generate_circle_points(0, 0, obs_R[i])
        rr.log(
            f"{path}/bounding_circle",
            rr.LineStrips2D([circle_pts.astype(np.float32)], colors=[[255, 255, 0, 128]]), # Yellow
            static=True,
        )

def log_timestep(
    k,
    x, y, th,
    w1, w2, w3, w4,
    path_pts,
    R_robot
):
    """Log dynamic elements for a single timestep k."""
    rr.set_time("step", sequence=k)

    # Log robot pose transform
    rr.log(
        "world/robot",
        rr.Transform3D(
            translation=[float(x), float(y), 0.0],
            rotation=rr.RotationAxisAngle((0, 0, 1), radians=float(th)),
        ),
    )

    # Log robot bounding circle (dynamic)
    robot_circle_pts = utils.generate_circle_points(0, 0, R_robot)
    rr.log(
        "world/robot/bounding_circle",
        rr.LineStrips2D([robot_circle_pts.astype(np.float32)], colors=[[0, 255, 255, 128]]), # Cyan
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