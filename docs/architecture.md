# Architecture

The model controls task execution and program-level improvement within the same persistent task. Runtime feedback, reusable knowledge, and recorded evidence connect these decisions.

## Agent

The Agent interprets the objective and observations, retrieves relevant Skills/Lessons/open improvements, requests execution, checks outcomes, and chooses revisions. Generic system instructions and the action schema belong in the core: they explain how the model uses these capabilities. Device-specific priors are loaded from an external configuration.

The improvement list belongs to the versioned knowledge system and is maintained by the Agent. It is linked to relevant skills and failures; it is not a separate hand-maintained developer TODO list.

## Runtime

Runtime executes the requested program and returns actual observations and execution receipts. It preserves request identity and distinguishes a rejected request from an execution with unknown results. The device implementation owns hardware-specific motion, sensing, and stop behavior.

Execution completion and task completion are separate. A robot command returning successfully does not prove that a key produced the intended character. The agent uses the subsequent observation to decide what happened.

## Skills and experience

- A Skill contains executable source, parameters, entry/exit conditions, limitations, and evidence references.
- A Lesson records a scoped finding or hypothesis and its evidence.
- An improvement record represents an unresolved opportunity or recurring failure, with priority and related knowledge.
- Demonstrations are optional source material. A supplied demonstration and an agent-authored skill derived from it have different provenance.

Knowledge can persist across normal tasks. Retrieval and explicit compaction expose relevant material while preserving version and evidence history. A candidate being saved is distinct from it being tested or promoted as stable.

## Records

Records preserve inputs, actions, programs, model-reported expectations, observations, outcomes, timing, usage, and source hashes. They support inspecting how a skill was revised and whether it was later executed. Model-internal reasoning is not published as evidence of physical understanding.

## Adapters

Applications supply task semantics such as sending, waiting, and checking a draft. Training environments can supply reset and evaluation independently of physical input. Demonstration readers supply inspectable source data. Hardware adapters supply the device behavior.

The core-only release excludes the reference machine's drivers, calibration, absolute poses, teleoperation collector, and deployment hints. The core-only candidate uses explicit external device contracts and passes isolated imports and portable integration checks. The private development tree is retained separately.

## Public integration seam

`runtime.device_adapter` declares the external device contract;
`runtime.program_runtime` executes complete bounded programs and records actual
receipts; `agent.runner` assembles these with the existing loop and knowledge
store. The default CLI is an authored offline repair/reuse fixture; continuous
service uses an explicit application/task contract and external callbacks.

Dedicated demonstration collection, trajectory readers and playback are outside
this core. Generic learning-source registration and evidence analysis remain so
that the model can derive device understanding from supplied examples. Fixed
interface facts belong to the external device description; learned scoped
findings belong to versioned Lessons/resources/profiles, linked from Skills and
open improvements. No private reference-device hints are shipped.

## Portability boundary

The factory seams accept another device's declared frame/views and another
model session. The supplied program surface remains observe, move one declared
active TCP, wait and stop; additional actuators or tools require an explicit
capability extension. Device drivers, calibration and application-specific
success conditions are supplied separately. The packaged continuous lifecycles
target typing/training and chat. The
[external-factory example](integration.md#runnable-external-factory-example)
tests a different synthetic waypoint task and view counts; it establishes the
software seam, not physical transfer or a general continuous-task protocol.
