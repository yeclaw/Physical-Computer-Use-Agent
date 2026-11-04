# Integration

The source package supplies the Agent/Runtime/knowledge/evidence framework and
offline fixtures. A new installation supplies its own device driver, calibrated
frames, camera capture, application behavior and model access. No reference
machine, prior keyboard map or account is required to import or test the core.

## Bring up a new setup

1. Implement the six device methods below in an external package. Make its
   `package.module:factory` importable in the same Python environment as the
   runner. The trusted operator selects factories, not the task model.
2. Declare the actual command frame, active tool/TCP and view IDs. Convert units
   and frames at this boundary; the base interface uses metres and unit xyzw
   quaternions. The SDK supplies measured state, completion and stop behavior.
3. Provide a model session factory, or explicitly select the optional Codex
   provider. Keep login state and credentials outside configuration, source
   control, observations and run artifacts.
4. Set `task`, `scope`, `knowledge`, `device` and `model` in the five-field JSON
   configuration. Relative paths resolve from the launching process's current
   directory, not from the JSON file's directory. Prefer explicit paths for a
   deployed service. Every run output directory must be new.
5. Use a new task/device/layout scope when applicability changes. Retaining the
   knowledge root preserves its audit history; it does not make another setup's
   poses or calibration valid. Scope separates knowledge namespaces, not OS
   filesystem permissions.
6. Run the offline example, then exercise the adapter's own observation,
   preflight and completed/unknown receipt paths before connecting a live model
   to hardware. The framework does not certify a new driver or physical setup.

## Device adapter

Implement `DeviceAdapter` in an external Python package. Select its trusted `package.module:factory` in configuration; the model cannot change factories. The supplied [MockDevice](../src/astra_robotics_agent/physical_keyboard/runtime/mock_device.py) is a complete, account-free implementation.

| Method | Contract |
|---|---|
| `describe()` | `AdapterDescription(id, coordinate_frame, views, evidence_mode)`; declare actual frame and camera IDs without moving hardware |
| `observe()` | `Observation(state, content_items)` with actual state/images; each image has an explicit declared `view_id` |
| `prepare(method, arguments)` | Pure validation; return `PreparedCall(method, arguments, token)` or raise `AdmissionRejected` before any effect |
| `dispatch(prepared)` | Execute once and return `CallReceipt`; an exception cannot establish what happened |
| `stop()` | Attempt stop and return its receipt; stopping is not proof of physical release |
| `close()` | Close owned resources and report failures |

The base bounded Python interface exposes `robot.observe()`, `robot.move_tcp(xyz_m, quaternion_xyzw)`, `robot.wait_bounded(seconds)` and `robot.stop()`. Positions are metres and quaternions are unit xyzw; the adapter names the actual command frame. The base call controls the adapter's declared active TCP; it has no implicit arm selector or gripper command. Other primitives need an explicit language/runtime capability extension and tests; implementing another method in the driver alone does not expose it to model-authored programs. Hardware initialization, transforms, planning and constraints remain in the adapter. No calibration, HOME, speed or workspace settings from the demonstrated physical robot are shipped. Synthetic example/reference-helper values are not a calibration for another device.

Receipts distinguish `(ok, completed)`, `(rejected, not_started)` and `(unknown, unknown)`. `completed` means the synchronous call boundary has completed with no pending execution, not merely queued. It does not mean a key registered or the task succeeded. After unknown dispatch, the runtime attempts stop, records stop failures, latches motion unavailable and never replays. A pure rejection or interpreter error after known completed calls can return to the model for a new plan; completed effects remain.

Images use `inputImage`, a JPEG/PNG data URL and a `view_id` in the adapter declaration. The runtime saves host timestamps, pixels and hashes without assuming three cameras. Host timestamps do not establish exposure synchronization. Historical images remain historical.

There may be zero, one or several declared views. Each supplied image must use
one of those IDs. Model sessions receive image URLs in content-item order; make
the image-index-to-view mapping explicit in observation text/state when views
could otherwise be ambiguous. Record actual sensor timestamps separately when
available. A command target, a queued acknowledgment and a measured pose are
different data; a real adapter must not manufacture observations from targets.

## Runner and model

The runner assembles the existing learning loop, runtime, knowledge and evidence registration. The JSON CLI is the simplest finite-task assembly; the Python API additionally accepts application/environment callbacks:

```python
from astra_robotics_agent.physical_keyboard.agent.runner import AgentRunner

runner = AgentRunner(
    session=my_session, device=my_device,
    knowledge="./knowledge", output="./runs/new-run",
    scope={"task_id": "typing", "device_id": "my-device", "layout_id": "layout-v1"},
    task="Enter the requested text and verify the actual result.",
)
report = runner.run()
```

The session provides `run_turn(prompt, callback, images=...)` returning `{turn_id, final_text, usage}`. `final_text` is one structured RSI decision. `usage=None` means unknown, never zero. The caller configures instructions/provider startup; `runner.run()` records cleanup and closes the session/device. The CLI handles assembly through factories.

Model factories accept `factory(settings, *, instructions, cwd, output)`; device factories accept `factory(settings)`. The model factory receives a dedicated provider output directory and an empty temporary working directory, and may create its log directory during construction. This working directory choice is not a filesystem read sandbox. Configure both in [examples/offline.json](../examples/offline.json). The supplied session is an authored offline fixture. To select the optional reference model factory explicitly, replace its model entry with:

```json
{
  "factory": "astra_robotics_agent.physical_keyboard.agent.providers:codex_factory",
  "settings": {"model": "gpt-6-astra", "effort": "xhigh", "service_tier": "default"}
}
```

The reference client uses Codex app-server. Access/authentication are supplied separately. Importing this factory does not start a model. Its wrapper was checked with a fake provider; source-export verification makes no live model calls. Inference-only client settings are not an OS sandbox. Other providers need their own validation.

Use a fresh run output; reuse the knowledge root and task/device/layout scope for ongoing learning. Change scope when applicability changes. The runner does not infer that old calibration remains valid on a new setup. Application policies and environment reset/evaluation are explicit contract adapters/callbacks to `runner.run(...)`, separate from control. Video capture is supplied externally; the core stores supplied evidence and recording indexes.

The core snapshots its own Python package. For reproducibility, separately
record the external adapter's code version, SDK/firmware version and calibration
identifier with the deployment. Store only reviewed, non-secret device facts in
observations or registered learning sources; do not dump an SDK configuration or
credential-bearing factory settings into evidence. The runner does not snapshot
an external package's complete source or infer its hardware configuration.

## Runnable external-factory example

The [external station module](../examples/external_station.py) and
[configuration](../examples/external_station.json) demonstrate the seam on a
different, entirely synthetic waypoint task. Its frame, geometry and view names
come from configuration. Its session consists of two authored decisions; its
one-pixel images are generated placeholders. It does not call a model, SDK,
camera, OS keyboard or physical device.

From the repository root, choose new output and knowledge paths:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:examples python3 \
  -m astra_robotics_agent.physical_keyboard.agent.runner \
  --config examples/external_station.json \
  --knowledge /tmp/physical-agent-external-knowledge-new \
  --output /tmp/physical-agent-external-run-new
```

The `examples` path makes the external factories importable; the example is
source-checkout material, not a preinstalled hardware plugin. After installing
the package, use `PYTHONPATH=examples physical-agent` with the same arguments.
For your implementation, replace the two factory names with your external
modules, retaining the documented input/output contracts. The example shows
configuration and evidence wiring, not how to operate a real robot.

The corresponding tests vary target geometry and use zero, one and two views,
verify separate layout scopes, exercise pure rejection and check that unknown
dispatch stops/latches the runtime without replay. This establishes software
portability at the seam, not physical transfer or autonomous learning.

## Learn device information from examples

Demonstrations are external learning material, not required executable trajectories. An independent collector can provide images, observed state/action pairs, timing and device descriptions through explicit `LearningSources` references. The core supports scoped registration, bounded inspection, analysis inputs, evidence links and versioned persistence. It does not collect or play demonstrations.

The agent can propose a device relationship, save it as a scoped Lesson or resource/profile, reference it from a Skill and revise it using new feedback. Direction/response or camera-to-contact findings should retain their source, frame, applicability, hypothesis/measurement distinction and validation limits. Learned device information is not automatically transferable calibration or permission to move another device.

No fixed six-category ICL taxonomy is required. Skills hold executable behavior and conditions; Lessons/resources hold findings and device understanding; improvements link unresolved opportunities to affected knowledge. They can accumulate and be revised or explicitly compacted during use. Model weights remain fixed.

## Boundaries

The candidate excludes the private SDK/controller, teleoperation, device prompts, trajectory catalog/playback, the former private historical-source RPC bridge and prior knowledge. The original implementation remains separate. Portable mock checks establish software contracts, not physical safety, cross-device adaptation or learning efficiency.

Authentication and device credentials are supplied by the deployer; see
[authentication and private evidence](privacy.md). The packaged training-reset
helper is a reference keyboard environment: its `home` alias is synthetic and
its state check expects `arms.left`. It is not a new device's calibration or
generic robot reset. A different environment supplies an explicit `reset`
callback with its own recovery and verification, or leaves reset unavailable.

## Continuous tasks and supplied sources

The offline CLI fixture intentionally finishes after one repair/reuse sequence.
A persistent knowledge root alone does not make it a continuous application.
For continuous task service, load an explicit task contract and supply the
application/environment callbacks in Python:

```python
from astra_robotics_agent.physical_keyboard.agent.campaign import load_task_contract
from astra_robotics_agent.physical_keyboard.adapters.skill_sources.learning_sources import LearningSources

sources = LearningSources(scope)
sources.register("reference-example", kind="human_demo", data={
    "observed_action": "supplied example", "observed_response": "measured response",
    "coordinate_frame": "your-declared-frame", "limitations": ["one example"]
})
contract = load_task_contract("continuous-task.json", scope)
report = runner.run(
    task_contract=contract, learning_sources=sources,
    reset=environment_reset, wait_for_update=application_wait,
    stop_requested=operator_stop,
)
```

These callbacks are externally supplied capabilities, not default device
operations. Use only those appropriate to the selected contract. The persistent
loop retains its normal completion/checkpoint/wait distinction; a phase summary
or individual successful reply does not terminate a continuous task. Fault
review/restoration is an explicit `fault_handler`/`assistance_wait` integration.
Unknown physical effects still cannot be retried by a model or restored merely
because an application task remains unfinished.

The packaged continuous contracts currently cover text entry/training and chat.
A different keyboard/layout can supply the corresponding new scope and task
data. A different activity with different completion semantics requires a
deliberate contract/application extension; changing an objective string alone
does not redefine success or add control capabilities. The finite external
waypoint fixture above does not claim to implement such a continuous contract.
