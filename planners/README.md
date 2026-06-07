# planners/

Each subdirectory is a submodule (one planner). Required files:

- `planner.py`: entry point. Subscribes to the SDK bridge, runs `step()`.
- `planner.yaml`: manifest: `action_type`, `obs_policy`, `depends`, `observations`.

  `action_type` is the planner's native action space and determines the `step()` return shape:
  - `differential_drive`: return `[v, omega]` (forward speed, yaw rate).
  - `omnidirectional`: return `[vx, vy]` or `[vx, vy, omega]` in the world frame (omega defaults to 0).

  The bridge dispatches per robot: holonomic robots receive `omnidirectional` actions verbatim; diff-drive robots receive the projection `(v=|vel|, omega=heading_err/step_dt + omega_in)` applied by [`bridge/projection.py`](../arena_planners/arena_planners/bridge/projection.py). Planners must not reinvent that projection inside `step()`.
- `package.xml`, `pyproject.toml`: ROS + Python packaging.
- `weights.yaml`: optional, HF-backed checkpoint manifest. Schema:

  ```yaml
  files:
    - repo: <hf-namespace>/<repo>     # e.g. arena-rosnav/drlvo
      filename: <name-on-hf>          # the asset's filename in the HF repo
      dest: <path-in-planner-dir>     # where to symlink locally, e.g. model/drl_vo.zip
      sha256: <hex>                   # optional integrity check
  ```

  `arena feature planners add <name>` reads this after submodule checkout and `hf_hub_download`s each entry into the HF cache, symlinking `dest` to the cached path. Missing `weights.yaml` is fine.
