# arena_social_cost_layer

Nav2 `costmap_2d::Layer` that paints social cost from `arena_peds` (`arena_people_msgs/Pedestrians`): a super-Gaussian personal-space disk around every pedestrian, widened for the `SURPRISED`/`CURIOUS` and `PANIC`/`THREATENING` animation states, plus a capsule along the segment between every pair of pedestrians sharing an active `interaction_id`, weighted by interaction type. Both terms stop at a hard radius (`personal_space_radius`, `group_extra_radius`), the sigmas set the falloff inside it and `falloff_shape` is the super-Gaussian exponent (1.0 is a plain Gaussian, larger values flatten the core and sharpen the edge).

Parameters under `<layer name>.`: `enabled`, `peds_topic`, `personal_space_radius`, `personal_space_sigma`, `group_extra_radius`, `group_sigma`, `falloff_shape`, `max_cost_value` (clamped below lethal), `stale_timeout` (seconds, older pedestrian messages are ignored), `cost_scale` and `cost_bias` (gain and floor on the normalized cost).

Robots on the Nav2 adapter load it into both costmaps with `robot.mobile.social_cost_layer:=true`; the parameter blocks live in `arena_robots/config/nav2/nav2.yaml`.

`cost_math.hpp` holds the pure cost functions, covered by `test/test_cost_math.cpp`:

```bash
arena build arena_social_cost_layer
arena test arena_social_cost_layer
```
