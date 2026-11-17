# frontier_vlm

A frontier vision-language model (Claude, GPT or Gemini) drives an Arena robot from an instruction
such as "Go to the laboratory". **The model never receives the goal pose or any coordinates.** It
looks at the camera image (and, at higher levels, a labelled floor plan), picks where to go next,
and the harness turns that choice into a collision-free path while the next call is already running.

It runs as a bridge planner with `goal_inputs: [instruction]`, like the trained VLN planners (CMA,
NaVid, NaVILA, StreamVLN), so the same scenarios and metrics apply to both.

## Levels (one added input each)

| level | config | the model gets | the model answers |
|---|---|---|---|
| L0 | `profile:=vln` | camera + instruction (the VLN planners' input) | 1-3 sub-goal pixels in the image |
| L1 | `floorplan:=on` | + labelled floor plan: rooms, doors, furniture, legend, robot and camera wedge | sub-goal pixels |
| L2 | L1 + `pipeline:=mission` | + one MISSION call: target room and a route of named rooms and doors | MISSION, then sub-goal pixels |
| L3 | L1 + `pipeline:=vlnav` | MISSION, then WAYPOINT calls: numbered candidates on the camera and on a local map (1 m rings, laser people tracks with motion arrows, the route), metres-to-go per candidate, a route-follower hint | a candidate number (or wait / turn / arrived) |

Room and door names come from the world's `world.yaml` zones; the MISSION and WAYPOINT answers use
enum schemas, so the model can only name places that exist.

## One decision

```
RGB + depth + pose (TF) + laser [+ arena_peds]
  -> prompt (+ floor plan / candidates by level), strict JSON schema, background thread
  -> answer: sub-goal pixels, or a candidate number
  -> pixel -> depth -> map point of the frame the model saw (answers stay valid while the robot moves)
  -> A* on a local grid (laser + static map), inflated by robot_radius
  -> waypoint chunk to the Arena bridge (or Nav2 DWB with robot.mobile.executor:=dwb)
  -> laser safety stop, which the model cannot override
```

Calls fire on events, not on a fixed clock: start of the episode, sub-goal or route waypoint
reached, plan blocked, a new moving person within `event_person_m`, plan older than `refresh_s`, and
**prefetch**: the next call starts when the time left on the plan is below the predicted latency
plus a margin, so the answer arrives about when the current plan runs out.

`latency_mode: real` (default) keeps driving during a call, as on a real robot. `latency_mode: zero`
makes `step()` wait for the answer; with `lockstep:=true` the simulator waits too, so model latency
costs no simulated time.

## Run

```bash
# CONTAINER
arena launch sim:=gazebo world:=hospital_1 headless:=true \
  robot:='jackal[camera=vln_rgbd]' robot.planner:=frontier_vlm robot.mobile.goal:=instruction \
  task.robots:=scenario task.obstacles:=scenario task.scenario.file:=fvlm_lab_crowd \
  robot.mobile.config.world:=hospital_1 robot.mobile.config.provider:=mock
```

- `provider:=mock` needs no key and answers every schema, for smoke tests.
- `provider:=claude|openai|gemini`, `model:=` ("" = provider default), `effort:=low|...|off`.
- Levels: `robot.mobile.config.profile:=vln`, `floorplan:=on`, `pipeline:=mission|vlnav`.
- Every key is documented in `planner.yaml`; override with `robot.mobile.config.<key>:=<value>`.
- Logs: one folder per episode under `log_dir` with `calls.jsonl` (prompt summary, answer, latency,
  cost), `ticks.jsonl` and the images sent to the model.

## API keys: host key proxy

The container never holds the real keys. `tools/keyproxy.py` runs on the host, adds the key to
each request and enforces a spend and request cap with a ledger:

```bash
# HOST: a .env with CLAUDE_API_KEY=..., OPENAI_API_KEY=..., GEMINI_API_KEY=... (chmod 600)
python3 tools/keyproxy.py --env .env --max-usd 10 --ledger ledger.jsonl    # 127.0.0.1:18765
curl -s 127.0.0.1:18765/_status                                            # spend per provider
```

In the container (host networking), point the providers at the proxy with dummy keys:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:18765 ANTHROPIC_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:18765/openai/v1 OPENAI_API_KEY=dummy
export GEMINI_BASE_URL=http://127.0.0.1:18765/gemini GEMINI_API_KEY=dummy
```

Cost in the ledger is an estimate from a price table; the providers' billing pages are authoritative.

## Files

| path | what |
|---|---|
| `planner.py`, `planner.yaml` | SDK glue and manifest (`waypoints`, `goal_inputs: [instruction]`, `signals: [arrived]`, all config keys) |
| `fvlm/harness.py` | the loop: call triggers, async calls, projection, A*, chunks, safety stop, logs |
| `fvlm/vlnav.py` | L1-L3: plan map rendering, route graph, mission route, people tracker, candidates, prompts |
| `fvlm/floorplan.py` | map.yaml + world.yaml zones and objects, local grid, A*, floor plan image |
| `fvlm/geometry.py`, `fvlm/perception.py` | camera model and projection; visible people, labels / inpainting |
| `fvlm/prompts.py`, `fvlm/providers.py` | prompts and schemas; Claude, OpenAI, Gemini and mock providers |
| `tools/keyproxy.py` | host key proxy with caps and ledger |
| `tools/hospital_sim.py`, `tools/offline_sim.py` | ROS-free simulators to try models and levels without Gazebo |
| `tools/render_episode.py`, `tools/render_pipeline.py`, `tools/render_vln.py` | episode videos from the logs (no GPU); `render_vln.py` works for any bridge planner from the SDK decision trace |
| `tools/cam_recorder.py`, `tools/peds_recorder.py` | record the recording camera and `arena_peds` next to a run |
| `tools/run_conditions.sh`, `tools/run_scenarios.sh`, `tools/run_vln_scenarios.sh` | batch runs: conditions, scenarios, or VLN planners side by side |
| `tools/check_scenarios.py`, `tools/scenario_peds.py`, `tools/plot_scenarios.py` | scenario checks, people replay for the offline sim, design figures |
| `scenarios/fvlm_*` | hospital_1 test scenarios (empty, crowd, pass, follow, sudden turn, cut-and-stop, group); `link.sh` links them into the world |

## Tests

```bash
python -m pytest tests -q      # no ROS, no network, no API
```

## Known limits

- L1-L3 have been run offline (`tools/hospital_sim.py`) and with the mock in the harness; not yet in Gazebo.
- The world needs zones (rooms, doors) in `world.yaml` for L1-L3; `hospital_1` has them.
- Without scipy the route graph uses a numpy distance field (same distances, slower).
