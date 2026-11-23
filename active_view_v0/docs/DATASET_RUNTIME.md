# ActiveView dataset runtime

This is the first dataset framework. It uses the existing Town03 traffic, ROI
and LiDAR configurations. The standard modes instantiate one UAV LiDAR,
one RSU, and the configured vehicle LiDARs. The legacy 25-point collector and
Where2comm oracle CLI continue to work on their existing data. New runs do not
use that candidate grid as an observation source.

## Install once per environment

In an existing editable checkout, `python -m active_view_v0.dataset_cli`
works immediately after updating that checkout. Run the following **once per
clone/worktree** to also get the `activeview` console command, without changing
existing dependencies. Substitute your checkout path.

```bash
conda activate carla-sim
python -m pip install --no-deps --no-build-isolation -e "$HOME/suyi/code/V2x_dataset/active_view_v0"
```

For a fresh environment install the project with
`python -m pip install -e "$HOME/suyi/code/V2x_dataset/active_view_v0[editor,test]"`
and install the matching CARLA 0.9.16 Python wheel once. Existing CARLA and
Where2comm environments can remain separate. For the optional detector smoke
test, use an environment that already has the matching CARLA Python API and the
project as well as OpenCOOD/PyTorch. No new runtime dependencies were added.

Set the data location once in your shell configuration if desired:

```bash
export ACTIVEVIEW_DATA_ROOT="$HOME/suyi/datasets/active_view_v1"
```

Without it, data goes to `datasets/` at the checkout root. `--data-root`
overrides the environment and also requires an absolute path. Commands can be
run from anywhere in an editable installation. Optional connection settings:
`ACTIVEVIEW_CARLA_HOST`, `ACTIVEVIEW_CARLA_PORT`, `ACTIVEVIEW_TM_PORT`. Keep
only one editor/preview/runtime/legacy CARLA tick client active at a time.

## Naming and storage

| Item | Convention |
|---|---|
| Scene source | `active_view_v0/scenes/Town03/scene_0001/config/scene.yaml` |
| Town ID | CARLA Town notation, e.g. `Town01`, `Town03`, `Town10HD`; map name in the base YAML may end in `_Opt` |
| Scene ID | `Town03/scene_0001`; give changed scene definitions a new ID |
| Family ID | lowercase, e.g. `town03_j1_j2`; related weather/traffic variants stay in the same split |
| Run path | `$ACTIVEVIEW_DATA_ROOT/Town03/scene_0001/<mode>/<run-id>/` |
| Run ID | lowercase and unique within a mode, e.g. `hover_001`; existing runs are never overwritten |
| Frame | `frames/000000/frame.json`, `frames/000000/agents/ego/lidar.bin`, etc. |
| LiDAR stream ID | `ego_lidar`, `cav2_lidar`, `rsu_lidar`, `uav_lidar`; file payload is little-endian float32 XYZI |

Save the brief next to the descriptor as `brief.yaml` for new scenes. The descriptor
references either a reusable YAML under `configs/` or a complete base YAML in its
own config directory. Overrides must use existing configuration keys. To add
a new kind of behavior, first implement and validate it in the scene engine.
Fields in the legacy single-junction `scenario` section do not drive the regional
engine; change `regional` for these scenes. The registered Town03 baseline
references the existing 40 s YAML **without changing it**. The draft experiment
`scene_0002` changes only J2's switch time from 20 s to 24 s. The Town03 reference
brief remains in `docs/scenes/`.

Every framework run records the frozen `effective_config.yaml`, scene hash,
source hashes, CARLA map fingerprint, and Git commit. Offline also writes
`manifest.jsonl` and frame payloads. Online writes `state_trace.jsonl`,
`events.jsonl`, and run metadata without writing raw sensor frames. GT is
logged by the runner and excluded from the policy input. This is a local Python
interface, not a sandbox for submitted third-party policy code.

## First acceptance run

1. Start CARLA and make sure no other script is ticking it. Validate the two YAMLs:

   ```bash
   activeview scenes
   activeview validate --scene Town03/scene_0001
   activeview validate --scene Town03/scene_0002
   ```

2. Check a short common prefix twice offline and once online. The default scene
   is 400 frames; `--frames 100` limits the run without modifying the scene.

   ```bash
   activeview run --scene Town03/scene_0001 --mode offline_hover --run-id hover_001 --frames 100
   activeview run --scene Town03/scene_0001 --mode offline_hover --run-id hover_002 --frames 100
   SCENE_DATA="$ACTIVEVIEW_DATA_ROOT/Town03/scene_0001"
   activeview run --config "$SCENE_DATA/offline_hover/hover_001/effective_config.yaml" --mode online_step --run-id hover_001 --frames 100
   activeview compare "$SCENE_DATA/offline_hover/hover_001" "$SCENE_DATA/offline_hover/hover_002"
   activeview compare "$SCENE_DATA/offline_hover/hover_001" "$SCENE_DATA/online_step/hover_001"
   ```

   Compare checks relative ticks, observed lamps, logical actors, vehicles,
   GT identities/geometry and sensor poses. Defaults are 5 cm and 0.5 degrees.
   Re-running with `--config` uses the saved YAML directly, without resolving today’s scene sources.
   Point-count differences are reported as diagnostics; no byte identity across
   hardware is claimed. Check the first difference before scaling collection.

3. Verify pause and motion with a deterministic +5 m offset along the CARLA
   **world** Y axis:

   ```bash
   activeview run --scene Town03/scene_0001 --mode online_step --policy offset --dx-m 0 --dy-m 5 --delay-ms 1000 --run-id teleport_001 --frames 30
   activeview run --scene Town03/scene_0001 --mode online_realtime --policy offset --dx-m 0 --dy-m 5 --delay-ms 150 --run-id delayed_001 --frames 100
   ```

   In `online_step`, the simulator waits for the policy at each decision tick;
   the returned offset applies before the next tick's capture. In
   `online_realtime`, tick pacing does not wait for the model. One strategy
   invocation can be active and one observation can wait; stale observations or
   actions are logged. Motion is kinematic with the scene speed limit (15 m/s
   for this scene), so travel is at most 1.5 m per 0.1 s tick, with no acceleration
   model. The tick rate, missing frames, timestamps and applied targets are
   logged; 10 Hz is a wall-clock target, not a throughput claim. Timing stamps
   use the host monotonic clock, and capture time is marked when the tick is submitted.

4. Validate the J2 change with a **full** run so both phase switches are
   observed. The 10-second check is too short.

   ```bash
   activeview run --scene Town03/scene_0001 --mode offline_hover --run-id full_001
   activeview run --scene Town03/scene_0002 --mode offline_hover --run-id full_001
   activeview check-run "$ACTIVEVIEW_DATA_ROOT/Town03/scene_0001/offline_hover/full_001"
   activeview check-run "$ACTIVEVIEW_DATA_ROOT/Town03/scene_0002/offline_hover/full_001"
   ```

   Inspect `first_late_phase_s` (expected 20.0 vs 24.0) and the health failures,
   if any. `activeview preview --scene Town03/scene_0001 --run-id bev_001` writes a
   BEV to the parallel `preview/` path using the legacy preview. Repeat for the
   variant and inspect the effect on queues and Ego motion before promotion.

## Plugging in inference

A caller can pass `--policy my_module:act` for an importable function
`act(observation) -> Decision(dx_m, dy_m)`. It receives live CARLA measurements
and transforms, not GT. Offsets are relative to the UAV at the sampled frame;
the target is then represented in world coordinates. `hover` and `offset`
are supplied for smoke tests. Adding both detector options wraps either
policy with the existing Where2comm forward at the decision rate:

```bash
activeview run --scene Town03/scene_0001 --mode online_step --policy hover \
  --model-dir "$MODEL_DIR" --opencood-root "$OPENCOOD_ROOT" --device cuda:0 \
  --frames 20 --run-id where2comm_001
```

`events.jsonl` records source tick, frame, inference timing, predictions and
the action application tick. This is a forward smoke test; AP at actual result
availability, full RGB/depth/segmentation rigs, policy training, and additional
map/route engines are follow-up work. Do not label an untested Town as approved
by copying a Town03 scene. Keep hardware validation results separate from unit tests.
