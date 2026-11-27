# Town01 · 11 条路线 / 15 个场景

每个目录的 `config/scene.yaml` 继承
[`town01_60s.yaml`](../../configs/town01_60s.yaml)；
`config/brief.yaml` 记录配对变量、路口、出生点、RSU、机位和 A/B 观察窗。
本组使用 `Town01`。统一 60 s、600 帧、10 Hz LiDAR、1 Hz UAV 决策；
Where2comm 继续用 80 m 名义 LiDAR 射程及车体坐标裁剪上界 `z_max=+1.0 m`。
同一数据划分组保留全部路线，避免共享路段和配对样本跨划分。

| 场景 | 主要对照 |
| --- | --- |
| 0001 / 0002 | 一辆同向移动车换成厢式车或轿车，其余车流和机位相同。 |
| 0004 / 0005 | Car2 同起点、同速，在 J26 分别留在主路或随 Ego 转弯，其余车流和机位相同。 |
| 0006 / 0007 | B 组同车数、同车速、同一路径，改出生站点压缩名义到达时段。 |
| 0008 / 0009 | B 组同一进口、同起点、同速，在 J139 分别转向 Ego 下游或相反方向。 |
| 0012 | B 目标点到最近的初始网格点约 214 m，独立运行连续飞行探针，不纳入网格主对照。 |
| 0013 | 初始悬停点已在 A/B 的名义射程内，保留为固定点对照。 |

## 服务器地图核对和预览

本地预检使用 [CARLA 公开 Town01 OpenDRIVE 样例](https://github.com/carla-simulator/opendrive-test-files/blob/master/OpenDrive/Town01.xodr)，
SHA-256：`15794f0dff5911837e6059ed628618ad666b3526748b698c0be25cdef7b27f70`。
地图名字匹配并不保证显式道路 ID 一致。先用服务器导出的地图文件复核；
车流交互、观察遮挡和收集后的检测结果需要在实机验证。

在 `active_view_v0/` 内，激活项目现有 CARLA Python 环境：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/check_lane_geometry.py /path/to/Town01.xodr --town Town01 --nominal-density

for i in 0001 0002 0004 0005 0013; do
  python -m active_view_v0.dataset_cli preview --scene "Town01/scene_$i" \
    --run-id "town01_check_$(date +%Y%m%d_%H%M%S)" || break
done
```

0012 启动后先悬停，再飞到 B。该探针验证轨迹与采集，不能作为自主策略结果：

```bash
python -m active_view_v0.dataset_cli run --scene Town01/scene_0012 --mode online_realtime \
  --policy scripts.town01_continuous_probe:b_waypoint --run-id town01_probe_0012
```
