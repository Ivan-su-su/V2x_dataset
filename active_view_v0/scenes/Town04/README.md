# Town04：主动视点候选场景

本目录是已经确认布局的 15 条 60 秒候选。每条有实际 CARLA 有向车道、车流 A/B、Ego /
Car2、RSU 世界坐标和固定 UAV 网格。先读 [场景要求](../../docs/SCENE_REQUIREMENTS.md)。

| 场景 | 主要机制 | 首轮试跑 |
|---|---|---|
| 0001–0003 | 十字错相、T 口、Ego 转向换轴 | 0003 |
| 0004–0006 | 连续路口、双 T 弧道、单向捷径 | 0006 |
| 0007–0009 | 绕街区、外环出入、Car2 离去 | |
| 0010–0012 | 主辅路交织、桥下出入口、环形下匝道 | 0011 |
| 0013–0015 | 上坡合流、低位换侧、外环分岔 | 0015 |

`config/scene.yaml` 可交给现有运行、采集入口；`config/brief.yaml` 记录每条的预期观察。
当前结果见 [验证记录](VALIDATION.md)。路口标志用于定位参考，桥下重叠位置不设虚假路口。
A/B 时段是车流规划窗口，实际事件时间由灯态和实测轨迹确认。

## 预检及预采

进入 `active_view_v0/`。可用 CARLA 服务器 `world.get_map().to_opendrive()` 导出
Town04 OpenDRIVE 底图后先离线检查。参数中的路径换成实际路径。

```bash
python scripts/check_lane_geometry.py /path/to/Town04.xodr --nominal-density
activeview preview --scene Town04/scene_0003 --run-id preview_001
activeview run --scene Town04/scene_0003 --mode offline_hover --run-id pilot_001
activeview check-run "$ACTIVEVIEW_DATA_ROOT/Town04/scene_0003/offline_hover/pilot_001"
```

用 `0006`、`0011`、`0012`、`0015` 依次重复。通过数量、运行进度和路线检查后，
再按 [采集及测评流程](../../docs/DATASET_RUNTIME.md#town03-60-s-lane-traffic-scenes)
使用各自的 `effective_config.yaml` 采 25 点和运行 Where2comm。按
[场景要求](../../docs/SCENE_REQUIREMENTS.md) 核对 A/B 的检测漏补数量和新 ID。
共享或相邻路线都留在同一 split group；不按场景编号随机拆分。

路网预检不承诺 traffic manager 实际选择了预定转向，也无法验证灯、遮挡、GT 密度或 AP。
先看真实 BEV、轨迹和逐帧 GT；尤其检查转弯、合分流及单车道匝道前后是否掉出目标带。
