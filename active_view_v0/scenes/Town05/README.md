# Town05：15 条基础场景与 2 个车流时序变体

每条以 `scene_XXXX/config/scene.yaml` 注册，和
[`town05_60s.yaml`](../../configs/town05_60s.yaml) 合成运行配置。
同目录的 `brief.yaml` 记录真实路口、Ego/Car2 起点及路线、RSU 杆位、UAV 地面投影和 A/B 车流。
统一 60 s、10 Hz、600 帧。离线采集同时保存 25 个 UAV 候选点，在线模式沿用现有单 UAV 接口。

| 场景 | 已确认布置 |
| --- | --- |
| 0001 | UAV 在 J1126 上方。 |
| 0002 | UAV 在 B 区路口附近，沿用确认图的地面锚点（J1722）。 |
| 0005 | Car2 从 J1427 和 J207 之间出发，朝 J207 行驶。 |
| 0007 | RSU 服务 A 区 J53，UAV 在 B 区 J979 上方。 |
| 0008 | RSU 服务 A 区 J1722，UAV 在 B 区 J979；Car2 在 J838 和 J979 之间，朝 J838 行驶。 |
| 0010–0012 | Ego 和 Car2 走同一路线，起点之间安排两辆同向背景车。 |
| 0016 | 0011 的对照变体：只向上游移动 J720 的 B 组三辆车，让它们更晚进入观测区。 |
| 0017 | 0013 的对照变体：只向上游移动 J2296 的 A 组两辆车，让早段来车在 ROI 内重叠更久。 |

其余点位沿用确认图；所有 Ego 路线不变。A/B 是车流区域和观察窗，路面 RGB 上的 J1/J2
是定位参考，不等同于每条的 A/B。预览和采集使用同一场景配置及 CARLA 原生交通优先级。
0001 的 B 区观察窗改为 44–60 s，仅更新 brief；车辆路线未改。0016 与 0011、0017 与 0013
各自使用同一场景家族和数据划分。基础配置的 Where2comm 评测高度上限为 `+1.0 m`，匹配本批使用的 checkpoint。

## 预览和采集

在 `active_view_v0/` 下运行。使用已有 CARLA 0.9.16 和采集环境；同一个 CARLA
实例先跑预览，再跑采集。

```bash
bash scripts/preview_town05.sh             # 15 条，总览取 0/15/30/45/59.9 s
bash scripts/preview_town05.sh 008 011     # 也可按编号选择
bash scripts/collect_town05_60s.sh          # 15 条，使用现有 25 点采集器
bash scripts/collect_town05_60s.sh 008 011  # 先采两条试验
bash scripts/preview_town05.sh 011 016 013 017     # 先对照预览两个变体
bash scripts/collect_town05_60s.sh 011 016 013 017  # 在同批次采集原版与变体
```

数据根目录由 `ACTIVEVIEW_DATA_ROOT` 设置，脚本默认
`/mnt/disk_4/suyi/datasets/active_view_v1`。预览图在
`Town05/scene_XXXX/preview/<RUN_ID>/global_bev_rgb_annotated_timeline.png`；
采集在 `Town05/grid_60s/<BATCH>/scene_XXXX/`，日志在批次的 `logs/`。
可以设置 `BATCH` 续跑已有批次，脚本跳过已完成的 600 帧；不覆盖未完成目录，
要补采时指定新批次。预览可设置 `RUN_ID`，须使用一个尚未存在的小写编号。

## 验证口径

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  python scripts/check_lane_geometry.py /path/to/Town05.xodr --town Town05 --nominal-density
```

几何预检可以确认有向车道、出入口接续、起点间距和 A/B 网格位置。实际通过信号灯、
Car2 跟车覆盖、目标密度，以及加入 Car2 或移动 UAV 带来的漏检恢复，仍须在 CARLA
实采和评测中确认。Town05 暂用同一 split group，避免重叠路线跨训练和测试拆分。
已完成的离线检查见 [验证记录](VALIDATION.md)。
