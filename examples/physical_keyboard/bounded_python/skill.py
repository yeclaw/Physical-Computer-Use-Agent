# This entire function/helper is candidate code, editable in a new frozen bundle.
def main(robot, profile):
    target = profile["target_xyz_m"]
    approach = [target[0], target[1], profile["approach_z_m"]]
    press = [target[0], target[1], profile["press_z_m"]]
    release = [target[0], target[1], profile["release_z_m"]]
    arrived = move_with_feedback(robot, approach, profile)
    if not arrived["reached"]:
        return {"attempt": "approach_not_reached", "needs_visual_check": True}
    try:
        pressed = move_with_feedback(robot, press, profile)
        robot.wait_bounded(profile["settle_s"])
    finally:
        released = move_with_feedback(robot, release, profile)
    return {"attempt": "executed", "press": pressed, "release": released,
            "needs_visual_check": True}
