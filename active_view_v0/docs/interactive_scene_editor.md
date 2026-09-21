# CARLA 城镇路口交互编辑器

这个工具独立于 `regional_preview`、`regional_collector` 和 `where2comm_eval`。
它连接正在运行的 CARLA，渲染真实 RGB 俯视画面，并在浏览器叠加车辆、
规划路线、实际轨迹、J1/J2、交通灯、RSU、无人机网格与 LiDAR 名义覆盖半径。
只有编辑器持有同步模式的 `world.tick()`；**运行时请先停止其他 CARLA
预览、采集或同步客户端**。

## 启动

在有 CARLA Python API 的 `activeair-sim` 环境，启动 CARLA 后运行：

```bash
cd active_view_v0
python -m pip install -e '.[editor]'
python -m active_view_v0.regional_editor \
  --config configs/dense_dynamic_town03_40s.yaml \
  --web-port 8765
```

CARLA 若从 Town10HD 切换到 Town03，首次载入可能较慢；进程启动后保持它运行。
浏览器访问 `http://127.0.0.1:8765/`。编辑器只监听服务器本机；
在另一台电脑使用时，通过 SSH 转发端口：

```bash
ssh -N -L 8765:127.0.0.1:8765 qsinghua_239
```

保持 SSH 会话运行，再在本机浏览器访问上述地址。CARLA 在 5090 主机上运行
时仍须具备可用的 CARLA 0.9.16、Python 环境和地图资源；网页客户端不要求
5090。请勿公开 8765 端口，该界面没有登录认证。

## 操作

- **选择地图和路口**：地图下拉框显示服务器已安装的 AirV2X 城镇地图；
  在全图点击“选 J1”和“选 J2”，或用“合法直行路线”下拉框选沿合法车道相连的
  一对路口。灰色的小字是地图路口编号，青色是当前 J1/J2。换地图后先保存并
  重跑，再在新地图上选路口。地图画面按全部路口范围取景，可用 `＋/－` 放大，
  在图内滚动查看局部。AirV2X 论文使用 Town01–04、Town06–07 和 Town12；
  实际可选项以 CARLA 服务器安装情况为准。
- **点选 Ego 起点**：在当前路线生效后使用工具，点击路线附近的道路；
  位置换算成 `regional.ego_start_advance_m`，需保存草稿并重跑。
  仍使用当前场景生成器定义的合法直行路线，暂不支持随意画转弯路线。
- **名称显示**：默认仅显示 Ego、协同采集车辆、RSU 和 Drone 名称；
  背景车与其他交通车只显示小点。勾选“其他车辆名称”可检查拥堵位置。
- **播放、暂停、步进 0.1 s / 1 s、重跑**：查看车辆是不是被信号灯、其他
  车辆或路线卡住。车辆列表显示即时速度；灯表显示期望状态和 CARLA 实际状态。
- **点选车辆**：查看车辆位置；勾选路线和实际轨迹来检查它走了哪条合法车道。
- **添加/编辑车辆**：先选 `j1_cross`、`j2_cross`、`j2_cross_reverse`、
  `corridor_forward`、`corridor_behind_ego` 或 `corridor_oncoming`，再设置
  `start_offset_m` 和 `speed_difference_pct`。放置位置是沿所选路线的距离，
  由现有场景生成器选择 CARLA 合法车道。车辆列表里的未生成提醒可能是出生点
  冲突。
- **点选 RSU 或 UAV 放置位置**：激活工具后点击画面，会将世界坐标换算为
  J1/J2 相对的前向/右向偏移。UAV 的点选移动整个 5×5 候选网格；
  `UAV 初始采样点` 在网格内选择初始悬停位置。编辑器只预览初始位置，
  不运行跟踪、巡逻策略或 Where2comm 推理。
- **保存草稿并重跑**：把改动写入
  `configs/dense_dynamic_town03_40s.editor.yaml`，然后从 t=0 重建。
  原始 `dense_dynamic_town03_40s.yaml` 不变，也不写现有数据集目录。
  可以把验收通过的草稿作为后续预览与采集的 `--config` 参数。

刷新网页会重新读取当前运行场景的配置；未点击保存的表单改动会丢失。
地图选择与路口切换会重建模拟场景，仍沿用当前两路口场景的车流和灯控配置；
在新的城镇中可能需要调整交通车辆的路线、出生距离和路口间距。
编辑器暂不支持三个以上路口的联合灯控或任意转弯；天气由 YAML 控制。
编辑器画的感知圈是量程/FOV 的几何边界，并不等于实际
点云可见目标或模型检测结果。

## 检查方式

先播放到 t=0、10、20、30 s，查看 Ego 轨迹、J1 双向车辆、J2 灯切换、
目标车流是否符合预期；有问题时暂停、改车流或偏移、保存草稿重跑。最后：

```bash
python -m active_view_v0.regional_preview \
  --config configs/dense_dynamic_town03_40s.editor.yaml --overwrite
```

停止编辑器后再运行预览/采集；它们都会操作同一台 CARLA 模拟器。
