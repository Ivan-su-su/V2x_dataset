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
| 0004 / 0005 | Car2 同起点、同速，在 J54 分别留在主路或随 Ego 转弯，其余车流和机位相同。 |
| 0006 / 0007 | B 组同车数、同车速、同一路径，改出生站点压缩名义到达时段。 |
| 0008 / 0009 | B 组同一进口、同起点、同速，在 J278 分别转向 Ego 下游或相反方向。 |
| 0012 | B 目标点到最近的初始网格点约 212 m，独立运行连续飞行探针，不纳入网格主对照。 |
| 0013 | 初始悬停点已在 A/B 的名义射程内，保留为固定点对照。 |

## 服务器地图核对和预览

当前配置从服务器运行中的 CARLA 0.9.16 导出的 `Town01_server.xodr` 生成和预检。
SHA-256：`97a7f6ac67812567e5c8ee0599cd823b23f80f30f3f97c502212e38b72e2b709`。
上一版使用的公开样例路口和连接车道编号不同；这一版按实际转向关系重新生成了车道序列和出生站点。
在其他 CARLA 安装中，请使用本机导出的文件复核。
车流交互、观察遮挡和收集后的检测结果需要在实机验证。

在 `active_view_v0/` 内，激活项目现有 CARLA Python 环境：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/check_lane_geometry.py "$HOME/Town01_server.xodr" --town Town01 --nominal-density

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
