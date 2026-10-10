# planners/

Each subdirectory is a submodule (one planner). Required files:

- `planner.py`: entry point. Subscribes to the SDK bridge, runs `step()`.
- `planner.yaml`: manifest: `action_type`, `rate_hz`, `depends`, `observations`, `config`, `goal_inputs` for VLA planners, `primitives` for discrete planners, and `chunk` for chunked planners.

  `config` is a free-form mapping the planner receives at Init as `planner_config`, through `main_loop(..., on_init=fn)` or `PlannerSDK.planner_config`. `robot.mobile.config.<key>:=<value>` overrides a key per launch.

  Each `Reset` carries `initial_state` with `goal_pose` (`x`, `y`, `theta` in the map frame). A `Reset` with `initial_state=None` starts an episode before its goal is dispatched.

  `step()` may return `arena_planners.sdk.Step(action, signal="arrived")` to send Arena a signal. A planner lists the signals it can send under `signals:` in `planner.yaml`. When a goto phase expects one, the reset's `initial_state` carries it as `signal`, and the phase ends when the planner sends it.

  `goal_inputs` (a subset of `[pose, instruction]`) marks a VLA planner, which runs under `robot.mobile:=vla` and only there. `robot.mobile.goal:=pose|instruction|pose+instruction` (default: all of `goal_inputs`) picks what a run reveals. `initial_state` then holds only the revealed keys, `instruction` being the goal's natural-language text (empty when the scenario gives none). Withholding `pose` also strips the bridge's `goal_pose` datasources and `SubgoalGenerator`s, and refuses planners with `depends.global_plan`.

  `action_type` is the planner's native action space and determines the `step()` return shape:
  - `differential_drive`: return `[v, omega]` (forward speed, yaw rate).
  - `omnidirectional`: return `[vx, vy]` or `[vx, vy, omega]` in the world frame (omega defaults to 0).
  - `discrete` (VLN-CE style): return `Step(command="forward"|"left"|"right")`, or `Step()` to hold still. Any other command or a plain list fails the step. `primitives: {forward_m: 0.25, turn_deg: 15.0}` (the defaults) sizes the moves, `Step(command=..., amount=x)` sizes one move as `x` meters forward or `x` degrees turned (0 means the primitive, a negative amount or a turn of 360 degrees or more fails the step), and the manifest must declare a `robot_pose` datasource of type `RobotPoseTFGenerator`. The bridge runs each move closed-loop on that pose at up to 0.5 m/s and 1 rad/s (lower when the robot's velocity limits are), and sends the next observation only once the move ends: on reaching the target within 2 cm or 1 degree, or at a timeout of three times the nominal duration plus 1 s, which a blocked move runs into.
  - `waypoints`: return `Step(chunk=[[x, y], ...])` (or `[x, y, yaw]` entries, yaw ignored), a path in meters in the robot frame of the observation it answers (x forward, y left). The bridge follows it with pure pursuit (lookahead `chunk.lookahead_m`, default 0.3 m, at `chunk.speed_mps`, default 0.5 m/s, capped by the robot's linear and angular limits, slowing near the last point, turning in place while its lookahead point lies more than 90 degrees off the heading) until `chunk.execute_s` (default 1 s) elapses or the robot is within 5 cm of the last point.
  - `velocity_chunk`: return `Step(chunk=[[v, omega], ...])`. The bridge applies each twist for `chunk.dt` (default 0.1 s, best a multiple of the planner tick) through the velocity-limit clamp, until `chunk.execute_s` elapses or the twists run out.

  Both chunk types need the `robot_pose` datasource like `discrete`, and `chunk: {execute_s, dt, lookahead_m, speed_mps}` (all optional, other keys refused) tunes them. `Step()` or an empty chunk holds still. A wrong entry width, a plain list, a `command`, an `amount` or an `action` vector fails the step, as does a `chunk` under any other action type. The bridge drives a chunk or move at 20 Hz (snapped to the physics step, never faster than the planner tick) between planner ticks, so a slow planner's robot does not run into its command timeout. It sends the next observation the tick a chunk ends and keeps its last twist meanwhile, so the robot keeps moving while the planner computes the next chunk. Cancel and reset stop it.

  `rate_hz` is the tick the policy was trained at (default 10); `robot.mobile.rate:=` overrides it.

  The bridge dispatches per robot: holonomic robots receive `omnidirectional` actions verbatim; diff-drive robots receive the projection `(v=|vel|, omega=heading_err/step_dt + omega_in)` applied by [`bridge/projection.py`](../arena_planners/arena_planners/bridge/projection.py). Planners must not reinvent that projection inside `step()`.
  `sensor_msgs/LaserScan` datasources always deliver the canonical scan: ray 0 along the robot heading, CCW over a full circle, non-returns and sub-`range_min` values equal to `range_max`, everything clipped to `[range_min, range_max]`. `params.canonical_beams` sets the ray count (mandatory for models with a fixed input width), otherwise the message's own count is kept. Planners never see a simulator's raw beam layout.
- `package.xml`, `pyproject.toml`: ROS + Python packaging.
- `weights.yaml`: optional checkpoint manifest. Every entry is either a Hugging Face file or a URL, and every entry needs `sha256`:

  ```yaml
  files:
    - repo: <hf-namespace>/<repo>     # e.g. arena-rosnav/drlvo
      filename: <path-in-hf-repo>
      revision: <commit-sha>          # optional, pins the HF revision
      dest: <path-in-planner-dir>     # e.g. model/drl_vo.zip
      sha256: <hex>
    - url: <https://... or gs://bucket/path>
      dest: <path-in-planner-dir>
      sha256: <hex>
  ```

  `arena feature planners add <name>` runs `python -m arena_planners fetch <name>` after checkout, which prints one line per file (dest, size, and `downloaded`, `verified` or `cached`). HF entries go through `hf_hub_download` into the HF cache. URL entries are streamed with urllib (`gs://b/p` as `https://storage.googleapis.com/b/p`) into a content cache at `$ARENA_WEIGHTS_CACHE`, else `arena-planners-weights/<sha256>` next to the HF hub cache. Each file's sha256 must match the manifest, otherwise the fetch fails naming both hashes and a bad URL download is removed. A `<sha256>.sha256-ok` marker in the content cache records a verified file, so later fetches skip re-hashing it (HF files get their marker under `hf/` there, keyed by blob path). `dest` is a symlink to the cached file, and a symlink pointing elsewhere is replaced. A `dest` whose content misses its `sha256` counts as missing in `check` and is replaced on the next fetch. Unknown keys, a missing `sha256`, or both `repo` and `url` in one entry fail the manifest. Missing `weights.yaml` is fine.
