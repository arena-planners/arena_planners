# arena_planners

Subprocess-isolated DRL planner bridge for Arena-Rosnav. Each planner runs in its own venv (own torch/gym/numpy versions), communicates with Arena over msgpack-on-ZMQ, and is driven by a single `step(features) -> [v, omega]` contract.

## Layout

```
arena_planners/
├── arena_planners/      # SDK: bridge, observations pipeline, resolver
├── planners/            # submodules, one per planner (drlvo, crowdnav, ...)
└── pyproject.toml
```

- The **SDK** lives in `arena_planners/arena_planners/`. Install with `pip install git+https://github.com/arena-planners/arena_planners.git`.
- Each **planner** is its own submodule under `planners/`, registered in this repo's `.gitmodules`. See [planners/README.md](planners/README.md) for the per-planner contract.

## Use from Arena

```sh
arena launch mobile:=drl mobile.planner:=drlvo
```

The `mobile:=drl` adapter (in Arena's `task_generator`) spawns the planner subprocess, pipes observations to it via the bridge, and routes the returned action to `cmd_vel`. Optional global plan via `mobile.global_planner:=nav2/navfn` (or any registered `<family>/<kind>`).

## Adding a planner

1. Create a submodule under `planners/<name>/` matching the contract in [planners/README.md](planners/README.md).
2. Register it in this repo's `.gitmodules` with a `planner = <name>` tag.
3. From Arena: `arena feature planners add <name>` initializes the submodule and fetches HF weights if `weights.yaml` is present.

## License

MIT. Per-planner submodules carry their own upstream licenses (drlvo: GPL-3.0, crowdnav: MIT).
