# Reproducible scenario workflow (v1)

## Dataset design facts and our adaptations

| Source | What the authors actually built | Rule we take for this project |
| --- | --- | --- |
| [AirV2X](https://arxiv.org/html/2506.19283v3) | 6.73 hours on CARLA Towns 01–04, 06–07 and 12; clear/cloudy/foggy/rainy and day/dusk/night; hover, patrol and escort UAV behavior | Vary **map, weather, lighting, traffic and UAV behavior** as separate scenario dimensions; label source + actor sensor types for every clip. |
| [Griffin](https://arxiv.org/html/2503.06983v2) | 255 clips (~15 s), four maps (Town03, Town06, Town07, Town10HD), multiple traffic densities and UAV altitudes, occlusion-aware labels and tracking | Keep continuous motion and stable target IDs; measure target visibility as well as object count. Griffin includes Town10HD even though AirV2X excludes Town10: do not equate their map selection rules. |
| [Adver-City](https://arxiv.org/html/2410.06380v2) | 110 scenarios in six weather conditions with CARLA/OpenCDA | Cross scene families with weather as controlled, recorded variants instead of modifying traffic by hand for each condition. |
| [OPV2V/OpenCOOD schema](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html) | Scenario → agent → timestamp layout | Retain stable scene ID, actor IDs, timestamps, transforms, effective config and train/val/test split at the **scenario family** level. |

These are methodological references, not claims of identical sensors or traffic rules. [CARLA 0.9.16 WeatherParameters](https://carla.readthedocs.io/en/0.9.16/python_api/#carlaweatherparameters) currently affects RGB rendering, **not** LiDAR physics or vehicle handling. Weather variants alone cannot validate LiDAR performance under precipitation. Dynamic sensor effects would need an explicit, separately versioned model.

## 1. Shared language: one brief before code

Copy `docs/scene_brief.template.yaml` to `docs/scenes/<scene_id>.brief.yaml` and fill it in. When describing a scene conversationally, the project owner can provide an annotated BEV and concise points:

- The **map and reference**: Town, junction IDs and world-coordinate x/y for J1/J2, road lane direction (CARLA yaw); distinguish the screen's left/right from **vehicle** forward/right.
- The **timed storyboard**: Ego start/route, checkpoints at t=0,10,20,30 s; exact signal phase intervals (including what happens at transition), number and direction of approaching/queued/moving vehicles; vehicle spacing and traffic legality.
- The **perception question**: which cooperative agents sense; which targets each viewpoint misses; RSU/UAV anchors, sensor ranges, Ego-centric ROI, duration and 10 Hz sampling/1 Hz decision; what evidence would refute the planned benefit.
- The **acceptance figures**: target min/mean, Ego displacement, signal states, group motion, occlusion duration, and model AP/recall reference policies. If a quantity is uncertain, write `null` and resolve via CARLA inspection before declaring the design approved.

The assistant restates this contract in a compact table, sketches the timeline or marks the annotated image, and flags only unresolved **material** contradictions. Once the user approves the design or requests direct implementation with enough detail, implement, validate and push without repeating approval questions. Save the final brief and a runnable versioned YAML to the same Git branch. An arbitrary map click or screen pixel is **not** a proven legal CARLA route.

The web editor writes a `*.editor.yaml` draft only. Once its lane routes, lights and checkpoints pass, promote that draft to a **new** versioned `configs/<scene_id>.yaml`, change the brief to `approved`/`validated` only after the corresponding checks, and commit both files. The next scene must start from that committed version, not from an untracked UI draft.

## 2. Minimal config set and ownership

| File | Committed? | Sole responsibility |
| --- | --- | --- |
| `configs/<scene_id>.yaml` | Yes | Full baseline scene (map/J1/J2, Ego route, roles, traffic, signals, seed, sensors, Ego ROI, UAV grid, clock). Existing reference is `dense_dynamic_town03_40s.yaml`; verify its motion before using it as a new scene's baseline. |
| `configs/weather/<condition>.yaml` | Yes, reusable across scenes | **Only** `carla.weather`: named CARLA preset plus optional explicit parameter overrides. |
| `configs/profiles/live_editor.yaml`, `snapshot_preview.yaml`, `offline_counterfactual.yaml` | Yes, reusable | Rendering/output toggles only; no traffic, signal, seed, sensor geometry or ROI changes. Editor is interactive inspection, snapshot preview renders key times, offline collector writes dense frames. |
| `docs/scenes/<scene_id>.brief.yaml` | Yes after agreement | Expected events and numeric checks for a specific scene; does not replace runnable YAML. |
| `configs/generated/<scene>__<weather>__<profile>.yaml` | No (reproducibly generated) | Frozen **full config** passed to an existing CLI. Stores original filenames/SHA256 in `recipe`; collector copies it to `<run>/effective_config.yaml`. |
| `<run>/manifest.jsonl`, `metadata.json`, `effective_config.yaml`, `collection_status.json`, `dataset_health.json` | No (data artifact) | Frame index, transforms, agent/target metadata, actual resolved config, collection completion and quality. Keep data outside Git. |

One scene baseline + reusable weather/profile overlays prevents full duplicate YAMLs. **Do not use the scene YAML directly for a weather experiment:** legacy preview/collector previously forced `ClearNoon`; after this patch all three entry points (`regional_editor`, `regional_preview`, `regional_collector`) read `carla.weather`. `regional_pipeline` runs collector **and proxy evaluation**; use `regional_collector` when only collecting.

To prepare a variant from the approved Town03 scene (from `active_view_v0/`):

```bash
python -m active_view_v0.scene_recipe \
  --scene configs/dense_dynamic_town03_40s.yaml \
  --weather configs/weather/clear_noon.yaml \
  --profile configs/profiles/live_editor.yaml \
  --run-name town03_j1j2_s42_clear_editor \
  --output configs/generated/town03_clear_editor.yaml
python -m active_view_v0.regional_editor \
  --config configs/generated/town03_clear_editor.yaml --web-port 8765
```

Close the editor before using the same CARLA instance for preview or collection. For a visual timeline:

```bash
python -m active_view_v0.scene_recipe \
  --scene configs/dense_dynamic_town03_40s.yaml \
  --weather configs/weather/rainy_noon.yaml \
  --profile configs/profiles/snapshot_preview.yaml \
  --run-name town03_j1j2_s42_rain_preview \
  --output configs/generated/town03_rain_preview.yaml
python -m active_view_v0.regional_preview \
  --config configs/generated/town03_rain_preview.yaml
```

For a 40 s / 400 frame offline counterfactual dataset:

```bash
python -m active_view_v0.scene_recipe \
  --scene configs/dense_dynamic_town03_40s.yaml \
  --weather configs/weather/clear_noon.yaml \
  --profile configs/profiles/offline_counterfactual.yaml \
  --run-name town03_j1j2_s42_clear_offline001 \
  --output configs/generated/town03_clear_offline001.yaml
python -m active_view_v0.regional_collector \
  --config configs/generated/town03_clear_offline001.yaml
```

Keep separate run names and **do not use `--overwrite` on approved data**. The existing collector saves a fixed UAV grid plus hover/tracking/patrol sensor modes in parallel; this is an offline counterfactual dataset, not an online policy rollout. For static-hover training choose a predeclared single position/stream, not the per-frame best position chosen from GT. Preview/collector use the same scene seed but should be compared by their recorded positions and trajectories; equal seed does not guarantee frame-identical Traffic Manager traffic across machines or CARLA versions.

## 3. Quality gates before publishing a scene

1. **Static code checks:** `python -m unittest discover -s tests -p 'test_scene_*.py'` and `python -m active_view_v0.scene_recipe ...` check config/clock/profile compatibility; verify `recipe` hashes.
2. **Live preview:** run one CARLA tick owner at a time, inspect t=0,10,20,30 and the switch instant ±0.1 s; capture real actor positions, J1/J2 actual lights, speed and displacement, legal route direction, support spawn count and visibility. `preview_motion_report.txt` should satisfy the brief. Do not infer vehicle movement from static BEV markers.
3. **Pilot recording:** collect one scene, check `collection_status.json` complete and expected frame count, `dataset_health.json` target min/mean and Ego motion, all required sensors and transforms, timestamps in `manifest.jsonl`, and the same Ego ROI/GT classes across variants. Missing required actors or stuck Ego invalidates the run.
4. **Detector smoke + benchmark:** run Where2comm `--smoke-test`, then full offline evaluation; report base, center hover, predeclared fixed point, tracking/patrol, best fixed **oracle bound** and speed-limited **oracle bound** separately. Report AP30/AP50 and R50 with scene counts and detector limitations; do not choose an oracle using held-out GT and present it as an online method.
5. **Split without leakage:** assign a complete `(map, junction route, traffic seed, traffic design)` family to train, val or test **before** crossing it with weather and UAV modes. Keep every weather variant and overlapping time window from the same family in the same split. Record CARLA version, Git commit, checkpoint/version and config hash. Treat reproducibility as measured by actual trajectory and sensor records, not guaranteed by the seed alone.

When changing map/junction outside the proven two-junction straight corridor, the current generator's traffic lights/vehicle routing may fail; create and validate a new scene engine or extend it explicitly. Avoid copying the Town03 required car positions into a new town and labeling a successful preview without seeing the actual trajectory.

## 4. First expansion after the Town03 reference

Pilot with **distinct route/seed families** first: for example three maps × two road layouts × two seeds = 12 families. Preview one scene from each family before multiplying by two or three weather variants; place every variant of a family in the same train/val/test split. Only after each family passes motion, GT density and detector smoke checks should a longer collection budget be fixed. This is a proposed project plan, not a dataset size claimed by AirV2X or Griffin.
