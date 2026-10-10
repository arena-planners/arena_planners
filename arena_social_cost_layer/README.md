# arena_social_cost_layer

Nav2 `costmap_2d::Layer` that paints social cost into the local costmap from
`arena_peds` (`arena_people_msgs/Pedestrians`), for the ANIM-on/ANIM-off
ablation between letting a robot perceive human animation/interaction
signals vs. treating humans as locomotion-only obstacles.

## Cost model

- **Personal space** (`cost_math.hpp::personalSpaceCost`): a Gaussian falloff
  around each pedestrian, sigma/amplitude widened for `PANIC`/`THREATENING`
  animation states vs. `IDLE`/`WALKING`/`RUNNING`.
- **HHI group/O-space** (`cost_math.hpp::groupBetweenCost`): for pedestrians
  sharing an ACTIVE `interaction_id` (bridged from humansim's
  `InteractionManager.active_membership` through `AgentState`/`Pedestrian`
  messages - see `humansim/arena_humansim/arena_humansim/core/agent_manager.py`
  and `task_generator/.../arena_humansim.py`), a Gaussian falloff off the
  *segment* connecting each pair of participants, not just off each endpoint,
  scaled by `interactionTypeMultiplier` (HUG/GROUP_CONVERSATION cost more
  than TALK_TO/WAVE_AT). This is what should push a planned path off
  "straight between the two people."

No goal-modifying logic lives here by design - HRI reactive behavior (e.g.
reorienting toward a human who points at the robot, see
`humansim/arena_humansim/config/scenarios/point_at_robot.yaml`) needs a
separate reactive controller/behavior node, a follow-on component.

## Wiring it in

This package only builds the plugin; it does not add itself to any nav2
bringup profile. See `config/social_cost_layer_params.yaml` for the
`local_costmap.plugins` snippet to splice into your robot's actual nav2
params (wherever `arena_robots`/your bringup profile defines them).

## ANIM on/off

Every `.py`/message field this layer reads goes to zero/absent in the
ANIM-off baseline (`humansim.anim_enabled:=false`, see
`humansim/arena_humansim`'s `anim_enabled` ROS param):
`Pedestrian.gestures` is emptied and `Pedestrian.interaction_id` stays `-1`
for everyone, since `AgentManager._process_interaction_scripts` never runs.
`animation_state` still reflects real locomotion (idle/walk/run), so the
personal-space term's baseline sigma/amplitude are unaffected - only the
danger-state widening and the HHI group term depend on the withheld signal.

## Testing

`cost_math.hpp` is pure functions (no ROS/costmap types), unit-tested in
`test/test_cost_math.cpp` via `ament_cmake_gtest`:

```bash
arena build arena_social_cost_layer
arena test arena_social_cost_layer
```

Manual/visual verification needs a live nav2 stack; see the top-level plan's
Verification section for the `arena launch` + `arena viz` walkthrough.
