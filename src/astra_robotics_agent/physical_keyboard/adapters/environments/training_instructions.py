"""Explicit training-editor reset documentation; no runtime side effects."""

TRAINING_RESET_INSTRUCTIONS = """When training_reset_available is true, you may explicitly request a training
reset when continued physical editing is not useful. It preserves all knowledge,
history and cumulative costs. You supply the release/return recovery program
and its intended return pose; the host does not invent a path. Recovery uses
the same robot interface, SDK protections and original receipts. For an existing
recovery Skill use recovery={"candidate_id":"saved ID"}; otherwise use the same
source_lines/profile/label fields as run_skill. Example shape (the recovery must
be replaced by your actual justified release/return program):
{"action":"reset","arguments":{"reason":"restart the training input", "return_pose":"home", "recovery":{"candidate_id":"saved recovery Skill ID"}}}
return_pose may instead be {"xyz_m":[...],"quaternion_xyzw":[...]} for your learned
observation position. home names the existing left command pose [0.125,0,0.225]
metres with identity quaternion, not a command to reinitialize or move both arms.
The framework runs your frozen recovery, checks measured arrival, asks the training
notepad to clear its input/IME and focus, and supplies fresh camera evidence.
Pose arrival alone does not prove physical release; inspect the new images.
Training clear is an environment privilege, never physical correction success.
Unknown recovery or reset outcomes stop without replay. If the tool is unavailable
or legacy viewing mode is active, reset is rejected without movement/clearing.
"""
