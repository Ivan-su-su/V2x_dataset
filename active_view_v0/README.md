# ActiveView-v1.3：20 秒、10 Hz、双向动态车流

主配置升级为 `configs/dense_dynamic_town03_20s.yaml`。v1.2 预览表明 Ego 在 10 秒
窗口内受红灯/前车排队影响而几乎没有位移，且随机背景车不能保证主走廊对向有车。
v1.3 因此不是简单延长录像：

- 采集 20 秒、严格 200 帧，UAV 仍每 1 秒决策；
- `ego_speed_difference_pct` 从 45 改为 0，即从限速的 55% 恢复到道路限速；
- 新增 5 辆确定性合法对向车，保留 6 辆同向前车和 2 辆后方驶入车辆；
- `preview_motion_report.txt` 报告每辆车的位移、采样路径长度和平均速度；
- `dataset_health.json` 额外要求 Ego 位移至少 40 m、平均速度至少 2 m/s。

先运行：

```bash
python -m active_view_v0.regional_preview \
  --config configs/dense_dynamic_town03_20s.yaml \
  --overwrite

cat /mnt/disk_4/suyi/active_airv2x/datasets/active_view_v0/dense_dynamic_town03_20s_v13_001/preview_motion_report.txt
```

确认 Ego 位移和双向车流后，再用相同配置运行 `regional_collector`。Where2comm 的
Base 名称、25 个候选名和三条连续 UAV 流均与 v1.2 完全一致。

---

# ActiveView-v1.2：10 秒、10 Hz 的稠密动态车路空场景

主配置为 `configs/dense_dynamic_town03_10s.yaml`。它生成连续 10 秒、严格 100 帧的
Town03 双路口片段，所有车辆均在记录开始前生成并由 Traffic Manager 沿合法车道持续
行驶，不使用中途传送或静止道具制造收益。

场景包含 2 辆感知车（Ego、CAV2）、1 个 RSU、14 辆确定性支撑车流和 14 辆背景车。
其中两辆正常行驶的大型车辆分别在 J2 横向车流和主走廊形成真实 ray-casting 遮挡；
评价目标仍按当前 Ego 坐标系的 AirV2X 风格范围筛选。

每帧保存以下接口：

| 类别 | 传感器名 | 数量 | 用途 |
|---|---|---:|---|
| 地面 Base | `ego_lidar`, `cav2_lidar`, `rsu_lidar` | 3 | Where2comm 固定基础输入 |
| 反事实 UAV | `uav_r0_c0` … `uav_r4_c4` | 25 | 最佳固定点和 detector oracle |
| 真实轨迹 UAV | `uav_mode_hover_lidar`, `uav_mode_tracking_lidar`, `uav_mode_patrol_lidar` | 3 | 三种可部署飞行基线 |

三种真实 UAV 都以 1 Hz 决策、10 Hz 连续插值移动：`hover` 固定悬停；`tracking`
只用 Ego 位姿跟踪其前方区域，不读 GT；`patrol` 按预设航路巡逻，不读交通状态。
最大速度为 15 m/s。25 个固定候选传感器不代表 25 架无人机，只用于同一时刻的反事实
视点比较。

## 1. 先预览交通布局

```bash
export ACTIVE_ROOT=/mnt/disk_4/suyi/active_airv2x
conda activate "$ACTIVE_ROOT/conda_envs/activeair-sim"
cd "$ACTIVE_ROOT/code/active_view_v0"
python -m pip install -e .

python -m active_view_v0.regional_preview \
  --config configs/dense_dynamic_town03_10s.yaml \
  --overwrite
```

确认俯视图中普通车辆均处于车道上，J1、走廊和 J2 均有连续车流，再进行正式采集。

## 2. 采集严格 100 帧

```bash
python -m active_view_v0.regional_collector \
  --config configs/dense_dynamic_town03_10s.yaml \
  --overwrite

export RUN_DIR="$ACTIVE_ROOT/datasets/active_view_v0/dense_dynamic_town03_10s_v12_001"
python - <<'PY'
import json, os
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
status = json.loads((run / "collection_status.json").read_text())
health = json.loads((run / "dataset_health.json").read_text())
print(status)
print("target min/mean:", health["minimum_target_count"], health["mean_target_count"])
PY
```

必须先看到 `keyframe_count: 100`。`dataset_health.json` 的密度门槛是每帧至少 18 个、
平均至少 20 个评价目标；若未通过，先检查实际俯视图和生成失败的车辆，不能直接解释
主动视点收益。

## 3. 用 AirV2X Where2comm 评测

```bash
conda activate "$ACTIVE_ROOT/conda_envs/activeair-det"
export OPENCOOD_ROOT="$ACTIVE_ROOT/code/Airv2x_gs"
export MODEL_DIR=/path/to/airv2x_intermediate_where2comm/release

# 先做 1 帧、1 个候选点的读数/权重/前向冒烟测试；同时检查三条真实 UAV 流。
CUDA_VISIBLE_DEVICES=6 python -m active_view_v0.where2comm_eval \
  --run-dir "$RUN_DIR" \
  --opencood-root "$OPENCOOD_ROOT" \
  --model-dir "$MODEL_DIR" \
  --device cuda:0 \
  --smoke-test \
  --overwrite

# 全量：100 ×（base + 25 个候选 + 3 种真实飞行模式）= 2900 次前向。
CUDA_VISIBLE_DEVICES=6 python -m active_view_v0.where2comm_eval \
  --run-dir "$RUN_DIR" \
  --opencood-root "$OPENCOOD_ROOT" \
  --model-dir "$MODEL_DIR" \
  --device cuda:0 \
  --overwrite
```

`where2comm_report.txt/json` 会同时报告 `base_only`、最佳固定点、逐帧上界、1 Hz 限速
oracle，以及 `uav_hover / uav_tracking / uav_patrol` 三条真实连续传感器轨迹。原有
`frame.json + manifest.jsonl + metadata.json` 结构、Base 名称和 25 个候选名称均未改变，
旧的几何评测和已有 Where2comm 评测目录仍可读取。

---

# ActiveView-v1.0：移动遮挡下的目标救回型主动 UAV

这一版的主配置是 `configs/active_occlusion_town03.yaml`：2 辆协同车、1 个 RSU、
1 架逻辑 UAV，以及普通交通目标。CAV-2 在 Town03 的 J2 横向道路上跟随一辆正常行驶的
大巴/卡车，前方两辆目标车被其遮挡。遮挡完全由 CARLA ray casting 产生，不使用人工
可见性掩码；所有车辆在记录前生成并沿合法车道连续运行。

Griffin 的无人机端是五相机、没有 LiDAR，因此本项目不能宣称 UAV LiDAR 参数复刻
Griffin。可对齐部分是车端 80 线、10 Hz、垂直 FoV `[-25°, 15°]` 和 25 m 高度子集；
UAV 80 m LiDAR 与 RSU 100 m LiDAR 是本项目自己的仿真设定。

## 1. 先看场景，不采集点云

```bash
export ACTIVE_ROOT=/mnt/disk_4/suyi/active_airv2x
conda activate "$ACTIVE_ROOT/conda_envs/activeair-sim"
cd "$ACTIVE_ROOT/code/active_view_v0"
python -m pip install -e .

python -m active_view_v0.regional_preview \
  --config configs/active_occlusion_town03.yaml \
  --overwrite
```

打开：

```text
$ACTIVE_ROOT/datasets/active_view_v0/active_occlusion_town03_v10_001/
├── global_bev_rgb_annotated_timeline.png
├── global_bev_sensor_ranges_timeline.png
└── scene_metadata.json
```

图中应同时看到 `CAV-2 / J2 transient`、`Moving bus / blocker`、两辆
`Occluded target`。如果大巴没有位于 CAV-2 与目标车之间，先调整
`car3_start_before_junction_m` 和三个 convoy 的 `start_offset_m`，不要直接正式采集。

## 2. 采集 30 秒并评测

```bash
python -m active_view_v0.regional_pipeline \
  --config configs/active_occlusion_town03.yaml \
  --duration-s 30 \
  --run-name active_occlusion_town03_v10_eval_001 \
  --overwrite
```

这会在 `t=0,1,...,30s` 保存 31 个关键帧。5×5 网格间隔 15 m、高度 25 m；
UAV 最大速度 15 m/s，因此 1 秒可移动一个横/纵格，不能一步走对角线。25 路 UAV
LiDAR 仍是同一时刻的反事实采集，不表示真实部署 25 架无人机。

主结果先看：

```text
occlusion_report.txt                 # 遮挡是否真实存在且 UAV 能看见
recovery_path_binary_metrics.txt     # 以“救回目标数”为目标求出的轨迹
recovery_report.txt                  # 最终 go / no-go 结论
```

`recovery_report.txt` 只有同时满足以下条件才输出 `ACTIVE_VIEW_PROMISING`：

- 早期和后期都存在足够的 Base 漏检及 UAV 可救回目标；
- 大巴确实处在 CAV-2 与前车之间，且 CARLA 点云证实 CAV-2 看不见；
- 从网格中心出发、受 15 m/s 约束的动态 UAV 比最佳固定点多救回至少 2 个目标实例；
- Recall 至少增加 1 个百分点，并产生至少 15 m 的有效移动。

连续点云密度结果仍保存在旧的 `oracle_results.json/path_binary_metrics.txt` 中，只作为
诊断；主结论使用 `recovery_oracle_results.json`，先最大化真实阈值救回数，再用点云质量
作极小的 tie-break，避免再次出现“点更密但 Recall 反而更差”。

---

# ActiveView-v0.9：稀疏车路覆盖下的主动UAV连续感知

`0.9` 新增一个不依赖 Ego-only 的主场景：3辆协同车、1个RSU、1架主动UAV，以及
普通目标车流。CAV-3在仿真早期经过J2，提供临时地面覆盖，随后沿合法路线离开；UAV的
5×5空域横跨连接道路与J2，测试它是否应该先保持在另一处稀疏区，再迁移到CAV-3留下的
覆盖缺口。

与旧场景相比有三个关键改变：

- 目标筛选使用固定在世界坐标中的双路口区域，Ego行驶不会带着评价范围一起漂移；
- CAV-3的进入/离开由连续路线产生，不在episode中途生成或删除Actor；
- 除连续点云质量分数外，新增路径级二值Recall和Miss Recovery Rate，只有真正跨过
  类别点数阈值才算救回目标。

## 1. 安装和预览场景

```bash
export ACTIVE_ROOT=/mnt/disk_4/suyi/active_airv2x
conda activate "$ACTIVE_ROOT/conda_envs/activeair-sim"
cd "$ACTIVE_ROOT/code/active_view_v0"
python -m pip install -e .

python -m active_view_v0.regional_preview \
  --config configs/sparse_coverage_town03.yaml \
  --overwrite
```

先查看：

```text
$ACTIVE_ROOT/datasets/active_view_v0/sparse_coverage_town03_v09_001/
├── global_bev_rgb_annotated_timeline.png
├── global_bev_sensor_ranges_timeline.png
├── global_bev_annotated_timeline.png
└── scene_metadata.json
```

黄色点划线矩形是固定区域ROI；紫色5×5网格向J1方向偏移15m；图中的
`CAV-3 / J2 transient` 应当在早期经过J2、后期离开。若路线符合预期，再正式采集。

## 2. 采集30秒并一键评测

```bash
python -m active_view_v0.regional_pipeline \
  --config configs/sparse_coverage_town03.yaml \
  --duration-s 30 \
  --run-name sparse_coverage_town03_v09_eval_001 \
  --overwrite
```

除16帧原始点云和25个同帧反事实UAV视点外，会生成：

```text
coverage_transition_report.txt   # CAV-3是否早期进入、后期离开J2服务区
path_binary_metrics.txt          # 固定/动态轨迹的Recall、救回数和MRR
idea_report.txt                  # 连续点云支持度的主动移动信号
active_view_validation.png       # 候选位置热图和轨迹
```

场景需要同时通过两类检查：

1. `coverage_transition_report.txt` 的 `Passed: True`，证明移动感知节点真的发生覆盖退出；
2. `path_binary_metrics.txt` 中动态轨迹相对 `best_fixed` 有正的
   `extra_rescued`、Recall和MRR，而不只是框内点数变多。

固定区域ROI为路向 `[-110,110]m`、横向 `[-60,60]m`，中心位于J1/J2中点。所有点云
后续仍可变换到当前Ego坐标系融合；固定ROI只决定哪些世界坐标目标参与评价。

## 3. 本场景的Agent与交通

| 类型 | 数量 | 是否提供协同点云 |
|---|---:|---|
| CAV-1 / Ego | 1 | 是 |
| CAV-2 / J1 | 1 | 是 |
| CAV-3 / J2 transient | 1 | 是 |
| J1 RSU | 1 | 是 |
| UAV | 1架；训练数据用25个反事实位置模拟 | 是 |
| 确定性普通目标车 | 最多5 | 否 |
| Traffic Manager背景车 | 目标12 | 否 |

CAV-3在J2前65m生成，预热3秒后进入记录，经过J2后沿额外140m路线离开。Ego降至约
限速35%，避免很快追到J2并立即填满CAV-3留下的缺口。主实验不增加第二个RSU；第二RSU
应当作为覆盖饱和度消融，而不是主配置。

# ActiveView-v0.8：同帧 Base-agent 消融

`0.8` 不重新采集，也不再调整交通场景。它在同一份 `0.7` 原始帧上，仅切换参与融合的
地面 LiDAR，回答“动态 UAV 的微弱收益是否被 RSU/CAV 覆盖掉”：

| profile | 参与融合的 Base 点云 | 目的 |
|---|---|---|
| `ego_only` | Ego | 检查单车感知下的主动 UAV 上限 |
| `ego_rsu` | Ego + RSU | 检查固定基础设施是否吃掉 UAV 收益 |
| `ego_cavs` | Ego + Car1 + Car3 | 检查车车协同是否吃掉 UAV 收益 |
| `full` | Ego + Car1 + Car3 + RSU | 原始完整协同基线 |

四组严格复用同一批帧、GT、Ego ROI、25 个 UAV 候选、阈值和类别权重。Car1/Car3 即使
从 Base 输入中移除，也仍在所有组中从 GT 排除，因此唯一自变量是地面传感器输入。

直接对已经采集完成的 v0.7 目录运行：

```bash
export ACTIVE_ROOT=/mnt/disk_4/suyi/active_airv2x
conda activate "$ACTIVE_ROOT/conda_envs/activeair-sim"
cd "$ACTIVE_ROOT/code/active_view_v0"
python -m pip install -e .

export RUN_DIR="$ACTIVE_ROOT/datasets/active_view_v0/regional_two_junction_v07_eval_001"
python -m active_view_v0.base_ablation --run-dir "$RUN_DIR"
```

如果该目录以前已跑过消融，需要明确覆盖结果时加 `--overwrite`。该命令不连接 CARLA，
不读取 GPU，也不修改 `frames/` 及 v0.7 顶层评测文件。新增输出为：

```text
base_ablations/
├── ego_only/ ego_rsu/ ego_cavs/ full/
│   ├── geometry_scores.npz
│   ├── geometry_details.json
│   ├── scoring_profile.json
│   ├── oracle_speed_only.json       # 只限制5m/s，不惩罚移动
│   └── oracle_penalized.json        # 限速并使用0.002/m移动惩罚
├── base_ablation_report.json
├── base_ablation_report.txt
├── base_ablation_summary.csv
├── base_ablation_summary.png
└── late_miss_diagnostics.json
```

先看 `speed_only_best_start` 相对各组自身 `best_fixed` 的增益。它允许为整段序列优化初始
点，但此后严格受 5m/s 速度约束，适合判断“存在物理可达的动态价值吗”。
`speed_only_center_start` 则回答“固定从 J2 网格中心起飞是否仍有收益”。如果无移动惩罚
时仍不到 3%，继续调惩罚系数没有意义。

- 只有 `ego_only/ego_rsu/ego_cavs` 明显、`full` 不明显：更诚实的故事是稀疏基础设施或
  agent dropout 下的主动 UAV，而不是完整车路空系统中的普遍动态增益。
- `full` 也达到 3% 以上：才值得继续做冻结检测器的 AP/Recall 验证。
- 四组都弱：应停止围绕该单一场景继续加车，不要用人工构造交通把差距“调”出来。

# ActiveView-v0.7：双阶段主动空中视点验证

`0.7` 针对上一轮真实结果修复场景，而不是调整传感器或评分来制造增益：

- Ego 在 J2 后增加 200m 近直行路线，避免 20 秒后随机左转；
- Ego 速度降低到限速的约 65%，让两个路口在 30 秒内始终处于有效时域；
- 五辆 Late 车辆在仿真开始前就生成，沿合法车道从 Ego 后方连续驶入；
- 保持 25m 高度、80m UAV LiDAR、5×5/10m 网格以及原评分不变；
- 新增 scene-health gate，分别检查早期和后期是否都存在 UAV 实际补回目标。

`0.6.1` 的稳定性修复仍保留：CARLA 采集放在独立子进程，并用批量命令销毁传感器和车辆，规避
CARLA 0.9.16 在大量附着式传感器退出时出现的 `trying to operate on a destroyed actor`
C++ 中止。即使异常发生在完整数据落盘之后，主进程也会核对 16 帧完整性并继续离线评测。

## 先直接跑这一条

确保 CARLA 正在 `127.0.0.1:2000` 运行，而且没有 ScenarioRunner、
`generate_traffic.py` 或第二个同步客户端调用 `world.tick()`：

```bash
export ACTIVE_ROOT=/mnt/disk_4/suyi/active_airv2x
conda activate "$ACTIVE_ROOT/conda_envs/activeair-sim"
cd "$ACTIVE_ROOT/code/active_view_v0"
python -m pip install -e .

python -m active_view_v0.regional_pipeline \
  --config configs/regional_two_junction_town03.yaml \
  --duration-s 30 \
  --run-name regional_two_junction_v07_eval_001 \
  --overwrite
```

该命令连续推进同一个 30 秒交通场景，在 `t=0,2,...,30s` 保存 16 个关键帧。
每个关键帧包含 Ego、Car1、Car3、RSU 和 25 个固定候选 UAV LiDAR，以及一张仅用于
场景核查的全局俯视 RGB。25 个 UAV 传感器是同一交通状态下的反事实视点，不代表部署
25 架飞机。

正式评分目标按当前 Ego 坐标系筛选：`x∈[-140.8,140.8]m`、
`y∈[-40,40]m`；无人机 5×5 网格仍固定在 J2 上空，不随 Ego 平移。

输出目录：

```text
/mnt/disk_4/suyi/active_airv2x/datasets/active_view_v0/regional_two_junction_v07_eval_001/
├── frames/                         # 16帧点云、全局RGB和GT
├── effective_config.yaml
├── metadata.json
├── manifest.jsonl
├── geometry_scores.npz             # [16,25]反事实位置分数
├── geometry_details.json           # 每个GT框的逐传感器点数
├── oracle_results.json             # 五种固定/动态策略
├── active_view_validation.png      # 分数热图与策略曲线
├── idea_report.json
└── idea_report.txt                 # 自动结论，先看这个
```

若采集完成、只想重新评测而不重跑 CARLA：

```bash
export RUN_DIR="$ACTIVE_ROOT/datasets/active_view_v0/regional_two_junction_v07_eval_001"
python -m active_view_v0.regional_pipeline --evaluate-only "$RUN_DIR"
```

判断重点是 `speed_constrained_center_start` 相对 `best_fixed` 的增益；前者从网格中心
出发、最大速度 5m/s，2秒最多移动10m。`per_frame_unconstrained` 可以瞬移，只是上界，
不能作为方法收益。报告中的分数是 LiDAR 落入 GT 框的连续支持度代理，不是 AP；只有代理
结果有明显差距，才值得进入冻结检测器的 AP/Recall 评测。

## 0.5：连续目标车流和各Agent实际传感器范围

旧版的人工横穿事件只保留作历史工程测试。新的主场景不再制造危险目标，改为：

- 两个由真实行驶车道连接的路口；
- J1上侧路边部署一个360° RSU；
- J2是无固定设施的UAV服务区域；
- Ego沿走廊从J1驶向J2，Car1横穿J1，Car3横穿J2；
- 三辆CAV与背景车辆全部由Traffic Manager连续驾驶；
- 5×5 UAV候选网格固定在J2，不跟随Ego；
- J2横向道路增加两辆小车，J1-J2走廊增加三辆延迟到达的小车；
- 五辆补充车在t=0即存在并合法行驶，不按时间瞬间生成；
- 单独绘制Ego、Car1、Car3、RSU和中心Drone的LiDAR范围；
- 先输出离线全局BEV验收场景，再接25路反事实LiDAR。

先列出Town03中真实道路相连的双路口候选：

```bash
cd "$ACTIVE_ROOT/code/active_view_v0"
python -m active_view_v0.regional_preview \
  --config configs/regional_two_junction_town03.yaml \
  --list-corridors
```

默认使用`corridor_rank: 0`。若布局不符合预期，在配置中换成其他rank，或者把输出的
`J1/J2`编号填入`junction_1_id`和`junction_2_id`。生成30秒连续场景和离线BEV：

```bash
python -m active_view_v0.regional_preview \
  --config configs/regional_two_junction_town03.yaml
```

如需重跑同名预览，加`--overwrite`。输出目录默认为：

```text
/mnt/disk_4/suyi/active_airv2x/datasets/active_view_v0/regional_two_junction_preview_001/
├── global_bev_annotated_timeline.png      # 自动包含三条完整路线的规划BEV
├── global_bev_sensor_ranges_timeline.png  # 各协同agent完整LiDAR范围
├── global_bev_rgb_annotated_timeline.png  # CARLA真实RGB + Ego/Car1/Car3/RSU/Drone/J1/J2
├── global_bev_rgb_raw_timeline.png        # 未叠加标记的CARLA真实RGB
├── global_bev_rgb_t000.0.png
├── global_bev_rgb_annotated_t000.0.png
├── global_bev_rgb_t010.0.png
├── global_bev_rgb_annotated_t010.0.png
├── global_bev_rgb_t020.0.png
├── global_bev_rgb_annotated_t020.0.png
├── global_bev_rgb_t030.0.png
├── global_bev_rgb_annotated_t030.0.png
├── scene_metadata.json
└── effective_config.yaml
```

这一命令不是正式数据采集：它只生成一台高空俯视相机和轻量场景快照，用来检查路口距离、
行驶方向、车辆是否卡住、RSU位置和J2的5×5 UAV空域是否合理。RGB上的标记由CARLA
相机内外参从世界坐标投影得到，不是按截图手工定位。RSU和Drone位置是虚拟传感器，CARLA
原始RGB中没有可见实体模型，因此在annotated文件中分别用红色P和紫色X显示。场景确认后，再把相同的
`RegionalIntersectionScenario`接到Vehicle/RSU/UAV LiDAR采集器。

范围图采用配置中的真实名义参数：Ego/Car1/Car3为80m水平360° LiDAR，RSU为100m
水平360° LiDAR；Drone只使用5×5中心位置。Drone高度25m、斜距80m，因此投影到地面的
最大半径约为`sqrt(80²-25²)=76m`。虚线表示无遮挡情况下的几何上限，建筑和车辆仍会造成
实际盲区。

这不是最终论文数据集，而是一个**先证伪、再扩展**的最小实验：在完全相同的动态路口状态下，同时采集 Vehicle、RSU 和 `5×5` 个候选 UAV LiDAR，回答下面的问题：

1. 每个时刻的最佳 UAV 位置是否真的变化？
2. 一个全程不动的最佳位置，是否明显弱于动态位置上界？
3. 加入无人机速度约束后，动态路径是否仍优于固定位置？

25 个 UAV LiDAR 是反事实采集装置，不代表真实部署 25 架无人机。正式方法阶段只保留一架可移动 UAV。

## 1. 当前版本做了什么

- CARLA 0.9.16 同步模式，固定 `0.1 s` 仿真步长。
- 自动排序并选择 Town03 路口，也可以手动指定 `junction_id`。
- 一个车辆穿越路口，两个被不同侧遮挡物遮住、时间错开的横穿目标。
- Vehicle LiDAR、固定 RSU LiDAR、路口上方 `5×5` UAV LiDAR 网格。
- 三路 RGB 场景预览：路口斜视、车辆前视、25 m 中心俯视。
- 网格间隔 `10 m`，高度 `25 m`，关键帧间隔 `2 s`。
- 保存每一关键帧的原始点云、世界坐标位姿、3D GT 框和完整配置。
- 用 GT 框内点数计算廉价几何可见性分数，再求固定与动态 oracle。

高度 `25 m` 参考 Griffin 的 `25±2 m` 子集。LiDAR 参数是首轮工程起点，不应在论文里表述成对 Griffin 硬件的复刻；Griffin 的公开仓库主要提供数据转换和训练代码，没有可直接调用的 CARLA 采集接口。

## 2. 安装到你的服务器

假设现有目录与环境是：

```bash
export ACTIVE_ROOT=/mnt/disk_4/suyi/active_airv2x
export CARLA_ROOT="$ACTIVE_ROOT/software/CARLA_0.9.16"

conda activate "$ACTIVE_ROOT/conda_envs/activeair-sim"
cd "$ACTIVE_ROOT/code/active_view_v0"
python -m pip install -e .
```

你前面已经验证 `import carla` 成功，因此通常不需要再配置。如果新终端里失败，再执行：

```bash
export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH"
python -c "import carla; print(carla.__version__)"
```

## 3. 启动 CARLA

继续使用已经验证过的离屏启动方式。注意 `r.GraphicsAdapter=6` 是 Vulkan 设备序号，不总等于 `nvidia-smi` 序号；最稳妥的是在启动进程前限制可见 GPU：

```bash
cd "$CARLA_ROOT"
CUDA_VISIBLE_DEVICES=6 ./CarlaUE4.sh \
  -RenderOffScreen \
  -quality-level=Epic \
  -carla-rpc-port=2000
```

采集脚本必须是唯一推进同步时钟的程序。ScenarioRunner、`generate_traffic.py` 和其他调用 `world.tick()` 的脚本要先停掉。只读取传感器、不推进时钟的 Web Viewer 可以保持运行。

## 4. 先选路口

```bash
cd "$ACTIVE_ROOT/code/active_view_v0"
python -m active_view_v0.inspect_junctions \
  --host 127.0.0.1 \
  --port 2000 \
  --map Town03 \
  --top 12 \
  --output junctions_town03.json
```

脚本优先选择道路数多、直行车道对多的路口。第一次可以让配置中的 `junction_id: null` 自动使用排名第一的路口。如果车辆或遮挡物出生碰撞，就把输出表中下一名的 id 填到 `configs/pilot_town03.yaml`。

## 5. 先跑 6 秒冒烟测试

冒烟测试临时改成 `3×3`，不会修改 YAML：

```bash
python -m active_view_v0.collector \
  --config configs/pilot_town03.yaml \
  --duration-s 6 \
  --grid-size 3 \
  --run-name smoke_3x3
```

终端应该连续打印关键帧、GT 数和传感器数。`3×3` 时传感器数应为 `14`：Vehicle + RSU + 9 个 UAV LiDAR + 3 个 RGB 预览相机。

采集开始后，在另一个服务器终端启动静态服务：

```bash
export SMOKE_DIR="$ACTIVE_ROOT/datasets/active_view_v0/smoke_3x3"
cd "$SMOKE_DIR"
python -m http.server 8081 --bind 127.0.0.1
```

在 VS Code 的“端口”面板转发 `8081`，然后本机浏览器打开：

```text
http://127.0.0.1:8081/scene_preview.html
```

页面会每秒检查最新关键帧并自动跟随，可直接查看车辆是否行驶、遮挡物是否挡住目标、横穿目标是否进入路口，以及 UAV 俯视范围是否正确。不要直接双击 HTML；浏览器的本地文件安全策略会阻止它读取 `manifest.jsonl`。

如果出现：

- `failed to spawn ...`：换一个 `junction_id`，不是重装 CARLA。
- `sensor ... skipped requested frame`：确认没有第二个同步客户端在调用 `world.tick()`。
- GPU/帧率压力过大：先把三个 LiDAR 的 `points_per_second` 各减半；不要先改网格和高度，否则对比条件会变。

## 6. 正式跑首个 5×5 pilot

```bash
python -m active_view_v0.collector \
  --config configs/pilot_town03.yaml
```

默认输出目录：

```text
/mnt/disk_4/suyi/active_airv2x/datasets/active_view_v0/town03_pilot_001/
├── effective_config.yaml
├── metadata.json
├── manifest.jsonl
├── scene_preview.html
└── frames/
    └── 000000/
        ├── frame.json
        ├── rgb/
        │   ├── overview_rgb.png
        │   ├── ego_front_rgb.png
        │   └── uav_center_rgb.png
        └── lidar/
            ├── vehicle_lidar.bin
            ├── rsu_lidar.bin
            └── uav_r0_c0.bin ... uav_r4_c4.bin
```

`.bin` 是 CARLA 原始 `float32 [x,y,z,intensity]`。`frame.json` 同时保存每个传感器的 `sensor→world` 矩阵和每个目标的 `actor→world` 矩阵，因此后处理不需要猜坐标系。

重复同一个 `run_name` 时脚本会拒绝覆盖。只有明确需要重跑时才加 `--overwrite`，它只删除该次 run 的目录。

## 7. 几何评分与轨迹验证

```bash
export RUN_DIR="$ACTIVE_ROOT/datasets/active_view_v0/town03_pilot_001"

python -m active_view_v0.scoring --run-dir "$RUN_DIR"
python -m active_view_v0.solve_oracle --run-dir "$RUN_DIR"
python -m active_view_v0.plot_results --run-dir "$RUN_DIR"
```

会生成：

- `geometry_scores.npz`：形状 `[T,25]` 的 score、coverage、gain。
- `geometry_details.json`：每个传感器对每个 GT 框的点数。
- `oracle_results.json`：五种路径与收益。
- `active_view_validation.png`：时空分数热图和路径曲线。

五种结果含义：

| 名称 | 含义 | 是否可实现 |
|---|---|---|
| `center_fixed` | 无人机固定在网格中心 | 是 |
| `best_fixed` | 事后选择全序列最好的一个固定点 | 是，但利用了测试集信息 |
| `per_frame_unconstrained` | 每帧瞬移到最佳点 | 否，只是理论上界 |
| `speed_constrained_center_start` | 从中心出发，速度不超过 5 m/s | 是，最公平的主对照 |
| `speed_constrained_best_start` | 初始点也由 oracle 选择 | 是理论上界，用于分析初始位置影响 |

从 v0.2 开始，主几何分数使用连续点云支持度，避免“超过阈值后 25 个位置全部同分”：

```text
q(n) = 1 - exp(-n / tau_class)
score = quality(Vehicle ∪ RSU ∪ UAV) + 0.5 × newly_improved_by_UAV
```

二值 `coverage/gain` 仍保留为诊断量，但不再作为 oracle 的主 score。它不是 AP，也不用于最终论文结论；它只负责回答“这个场景有没有动态选点信号”。并列最优时，逐帧 oracle 优先保持原位置，不再把 `argmax` 的索引变化误报成无人机移动。

## 8. 什么时候值得进入检测模型阶段

先看 `oracle_results.json`，至少应满足：

1. `per_frame_unconstrained.switch_count >= 2`，最佳位置不是恒定的。
2. `best_fixed` 明显弱于 `per_frame_unconstrained`，否则动态选点没有空间。
3. `speed_constrained_center_start` 仍优于 `center_fixed`，否则收益依赖瞬移。
4. 轨迹每 2 秒最多走一个横向或纵向网格，不能对角跳 14.14 m。

如果不满足，不要马上训练检测器。先从 `geometry_details.json` 判断是场景没有形成互补、RSU 已覆盖全部目标，还是空中 LiDAR 太稀疏，然后只调整场景事件或传感器参数中的一个因素。

通过这一步后，再把几何 score 替换成固定检测器在 25 个位置上的同帧 AP/Recall，并保持 oracle 与评测代码不变。这样能把“场景是否成立”和“网络是否训好”两个问题分开。

## 9. 当前版本的边界

- 无人机位置由固定 LiDAR 传感器模拟，没有飞行动力学、机体遮挡、抖动和能耗。
- 只实现 LiDAR 几何预验证，尚未接入 OpenCOOD / Where2comm 检测输出。
- RSU 与 UAV 参数是首轮可运行配置，正式数据集前需要做参数消融和论文来源复核。
- 自动生成的遮挡场景用于验证假设，不应直接当作最终 benchmark 的全部场景。

## 10. 本地测试

```bash
cd "$ACTIVE_ROOT/code/active_view_v0"
python -m pytest -q
```

测试覆盖坐标变换、5×5 网格中心、固定 oracle、动态规划以及速度约束不能瞬移。
# AirV2X Where2comm checkpoint evaluation

The geometry score is a scene-design diagnostic, not detector AP.  To run the
AirV2X LiDAR Where2comm checkpoint directly on a collected CARLA episode:

Create the isolated detector environment (do not install the legacy OpenCOOD
requirements into the CARLA environment):

```bash
export ACTIVE_ROOT=/mnt/disk_4/suyi/active_airv2x
cd "$ACTIVE_ROOT/code/active_view_v0"
bash scripts/setup_where2comm_smoke.sh
```

```bash
conda activate /mnt/disk_4/suyi/active_airv2x/conda_envs/activeair-det
cd /mnt/disk_4/suyi/active_airv2x/code/active_view_v0

python -m active_view_v0.where2comm_eval \
  --run-dir /mnt/disk_4/suyi/active_airv2x/datasets/active_view_v0/active_occlusion_town03_v11_001 \
  --opencood-root /mnt/disk_4/suyi/active_airv2x/code/Airv2x_gs \
  --model-dir /path/to/airv2x_intermediate_where2comm_checkpoint \
  --device cuda:6 \
  --smoke-test
```

The model directory must contain the exact Hugging Face `config.yaml` and
`net_epoch16.pth`. A passing smoke test prints `Dataset read OK`,
`Checkpoint load OK`, `Model forward OK`, and finally `SMOKE TEST PASSED`.
It evaluates one frame with base agents and one UAV candidate. Results are cached
under `<run-dir>/where2comm_eval`, so interrupted full-grid inference can be
resumed by running the same command without `--overwrite`.  The runner refuses
a partial checkpoint load by default.  Do not use
`--allow-partial-checkpoint` for reported experiments.

The first report is class-agnostic BEV AP because the CARLA pilot currently
contains a narrower class set than AirV2X.  All point clouds and GT are converted
to the current Ego-LiDAR frame and clipped to the checkpoint's native range.
