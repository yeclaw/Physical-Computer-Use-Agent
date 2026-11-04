# Authored synthetic example; not learned geometry or approved physical limits.
def move_with_feedback(robot, target, profile):
    for iteration in range(profile["max_steps"]):
        observation = robot.observe()["observation"]
        actual = observation["pose"]["xyz_m"]
        error = [target[0] - actual[0], target[1] - actual[1], target[2] - actual[2]]
        largest = max(abs(error[0]), abs(error[1]), abs(error[2]))
        if largest <= profile["position_tolerance_m"]:
            return {"reached": True, "iterations": iteration, "actual_xyz_m": actual}
        scale = min(1.0, profile["max_step_m"] / largest)
        next_pose = [0.0, 0.0, 0.0]
        for axis in range(3):
            next_pose[axis] = actual[axis] + scale * error[axis]
        robot.move_tcp(next_pose, [0, 0, 0, 1])
    return {"reached": False, "reason": "bounded_motion_not_reached"}
