# Town06：15 个场景及 Car2 布置

场景由各目录的 `config/scene.yaml` 注册，继承
[`town06_60s.yaml`](../../configs/town06_60s.yaml)。`config/brief.yaml` 记录
Ego 和 Car2 的车道路由、RSU 杆位、UAV 起点、A/B 事件窗和目标角色。
统一 60 s / 600 帧、10 Hz LiDAR、1 Hz UAV 决策、80 m 名义射程；
Where2comm 的 Ego 坐标裁剪上界保持 `z_max=+1.0 m`。
所有场景先放在同一数据划分组，避免重复/相邻路线跨划分。

## Car2 修订和对照

| 场景 | Car2 设置 | 在 Ego ROI 且距 Ego ≤80 m，原草案 → 当前（秒） |
| --- | --- | ---: |
| 0001 / 0002 | 与 Ego 同向，沿路线从后方约 28 m 出发；厢式车/轿车配对共用路线和交通时序。 | 0 → 60 |
| 0003 / 0004 | 原有对照：同向出发，Car2 在 J452 分别驶离和随 Ego 转向。 | 保留：25.8 / 60 |
| 0005 / 0006 | Car2 从后方约 29 m 出发，与 Ego 一起转弯；B 组同路线/车速，比较分批和集中到达。 | 0 → 60 |
| 0007 / 0008 | Car2 从后方约 32 m 出发，一起走 J1050–J1203 匝道；后段比较主线与匝道来车。 | 17.5 → 60 |

表中的时间来自向前沿车道按目标速度走的几何估算。Car2 在 Ego ROI 内只说明空间关系；
是否能覆盖 Ego 漏掉的目标，需要测真实轨迹、遮挡和检测结果。0009–0015 使用已审路线
和原 Car2 布置，其中 0015 的 `continuous_extension` 是独立于网格内 B 的飞行检查点。

## 先核对地图，再预览

本地预检使用 OpenDRIVE SHA-256
`a90b613175fa837fdc128ddfaf5ddad25797b128fdbd8cd945974bc21a60496e`。
另一版 Town06 文件的路口编号不同。先用服务器实际 Town06 导出的地图运行检查；
哈希不一致时重点核对车道、路口连接和机位。
服务器 CARLA 预览、车辆跟车/等待、遮挡和检测收益尚未验收。

在 `active_view_v0/` 内，激活现有 CARLA Python 环境后：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# 使用从目标 CARLA 实例导出的 Town06.xodr；此命令不启动仿真。
python scripts/check_lane_geometry.py /path/to/Town06.xodr --town Town06 --nominal-density

# 启动 CARLA 后，先对两组路口通道和一组匝道预览。
for i in 0001 0003 0004 0005 0007; do
  python -m active_view_v0.dataset_cli preview --scene "Town06/scene_$i" \
    --run-id "town06_check_$(date +%Y%m%d_%H%M%S)"
done
```

完整采集前检查实际 Car2 在 A/B 窗的位置、车距和观测，0003/0004 是否按规定分流，
以及 GT 密度和 B 窗的新目标是否合格。使用已有采集和 Where2comm 命令；
按中心悬停、最佳固定点、限速移动分别报告 AP50/R50。详见
[场景要求](../../docs/SCENE_REQUIREMENTS.md)。
