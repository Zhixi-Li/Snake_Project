# Change Log

## 2026-05-31 Velocity Tracking Tuning

### Motivation
- The initial play plots showed weak command tracking: actual Virtual Chassis vx/vy did not stay close to the commanded lines, and wz had visible oscillation even though the command yaw rate is zero.
- The MuJoCo evaluation grid focuses on vx in [-0.2, 0.2] and vy in [-0.1, 0.1], so the training distribution should emphasize that region before broader robustness tuning.
- GPU memory still has headroom, so the PPO mini-batch setup was adjusted to better support higher parallel environment counts such as --num_envs 8192.

### Modified Files
- source/snake_project/snake_project/tasks/manager_based/velocity_tracking/velocity_env_cfg.py
- source/snake_project/snake_project/tasks/manager_based/velocity_tracking/config/Snake_7DOF/agents/rsl_rl_ppo_cfg.py

### Environment / Command Changes
- Narrowed the training velocity command range from vx [-0.4, 0.4], vy [-0.2, 0.2] to vx [-0.25, 0.25], vy [-0.12, 0.12].
- Kept ang_vel_z fixed at 0.0, matching the evaluation setup.
- This concentrates early training capacity on the final evaluation region while keeping a small margin beyond the exact test grid.

### Reward Changes
- Increased Virtual Chassis linear velocity tracking weight from 5.0 to 6.0.
- Increased Virtual Chassis yaw-rate tracking weight from 1.0 to 1.5 to suppress unwanted wz drift.
- Increased joint acceleration penalty from -2.5e-7 to -5.0e-7.
- Increased raw action-rate penalty from -0.01 to -0.015.
- Left policy observations and action scale unchanged to preserve compatibility with sim2sim_eval.py observation construction and avoid changing the action interface before validating this tuning round.

### Curriculum Changes
- Enabled command velocity curriculum.
- Set min_curriculum to 0.05, max_curriculum to 0.25, step_size to 0.025, threshold_ratio to 0.75.
- The intended behavior is to learn stable low-speed tracking first, then expand toward the evaluation command range.

### PPO / Throughput Changes
- Increased max_iterations from 5000 to 10000 for a stronger baseline training run.
- Increased num_mini_batches from 4 to 8 so larger rollouts from higher --num_envs values keep more manageable mini-batch sizes.
- Recommended next run: start with --num_envs 8192 if GPU memory allows, then compare FPS and final sim2sim MAE against the 4096-env baseline.

### PhysX Buffer Follow-up
- Increased `self.sim.physx.gpu_max_rigid_patch_count` from `10 * 2**15` to `16 * 2**15`.
- This addresses the PhysX `Patch buffer overflow` observed when running with higher parallelism. The previous value was 327680, while the runtime error requested at least 398073; the new value is 524288.
- If the same error appears again at even larger `--num_envs`, either increase this buffer further or reduce `--num_envs`.

### Next Validation Steps
- Train a new policy with the updated config, preferably with --num_envs 8192 and --headless.
- Run play.py with --manual_command --plot for representative commands such as (0.2, 0.0), (0.0, 0.1), and (-0.2, -0.1).
- Export the JIT policy through play.py and run sim2sim/sim2sim_eval.py.
- If IsaacLab tracking improves but MuJoCo MAE remains poor, add light domain randomization in a later round rather than enabling strong randomization immediately.

## 2026-06-01 运动约束奖励与可读诊断输出

### 修改动机
- 新一轮训练仍然没有形成稳定的二维平面速度跟踪，主要现象是 Virtual Chassis 的 vx/vy 不能稳定贴合指令，wz 仍有漂移。
- 蛇形机器人自由度冗余较高，单纯依赖速度跟踪奖励容易学到“会摆动但不按指令走”的策略，因此增加软约束来引导更像二维平面运动的步态。
- TensorBoard 能看训练趋势，但不够直观地回答“速度误差是多少、wz 是否被压住、动作是否过激”。因此增加 CSV 和 Markdown 诊断文件，便于训练中和测试后快速比较不同实验。

### 奖励函数新增内容
- 在 `mdp/rewards.py` 中新增 `joint_curvature_l2`：惩罚相邻关节角度的二阶差分，减少局部尖锐折弯，让身体曲率更平滑。
- 新增 `joint_mean_bend_l2`：惩罚所有 yaw 关节长期同向偏置，减少整体 C 形弯曲导致的无效转向。
- 新增 `VirtualChassisLateralVelocityPenalty`：当有平面速度指令时惩罚实际速度中垂直于指令方向的分量；当指令接近 0 时惩罚整体平面速度，减少零速指令下乱动。
- 新增 `VirtualChassisYawRateAbsPenalty`：直接惩罚 Virtual Chassis 的绝对 yaw rate，辅助已有的 `track_ang_vel_z_exp` 压制 wz 漂移。

### 奖励权重调整
- 在 `velocity_env_cfg.py` 中加入 `vc_lateral_velocity_l2`，权重为 `-1.0`。
- 加入 `vc_yaw_rate_abs`，权重为 `-0.2`。
- 加入 `joint_curvature_l2`，权重为 `-0.04`。
- 加入 `joint_mean_bend_l2`，权重为 `-0.05`。
- 将 `joint_amplitude` 从 `0.2` 降到 `0.15`，将 `phase_propagation` 从 `0.4` 降到 `0.3`，将 `motion_coordination` 从 `-0.5` 放松到 `-0.4`。这样做是为了避免辅助步态奖励压过速度跟踪目标。

### 训练诊断输出
- 在 `scripts/rsl_rl/train.py` 中新增 `--diagnostics_interval` 参数，默认每 200 个环境 step 写一次诊断。
- 新增输出文件：`logs/rsl_rl/<experiment>/<run>/diagnostics/train_monitor.csv`。
- CSV 记录字段包括 reward 均值/方差、done 比例、指令速度均值、Virtual Chassis 实际速度均值、wz 绝对值、xy 速度误差、动作幅度、关节位置/速度幅度。
- 该文件用于快速判断训练是否正在向“速度误差下降、wz 下降、动作不过激”的方向发展。

### 测试诊断输出
- 在 `scripts/rsl_rl/play.py --plot` 中新增逐步 CSV 和 Markdown 摘要。
- 新增 `plots/play/play_timeseries.csv`：逐时刻保存 cmd/act 速度、误差、planar error、动作幅度和关节幅度。
- 新增 `plots/play/diagnostics_summary.md`：保存 warmup 后的 vx/vy/wz MAE、planar MAE、RMSE、实际速度均值/标准差、动作和关节幅度。
- 后续比较 checkpoint 时，优先看 `diagnostics_summary.md` 中的 `mae_planar`、`mae_wz`、`act_wz_abs_mean` 和动作/关节幅度。

### 后续建议
- 如果新增约束后机器人明显“不敢动”，优先减小 `vc_lateral_velocity_l2` 或 `joint_curvature_l2` 的惩罚权重，而不是继续增加速度跟踪奖励。
- 如果 vx/vy 变好但 wz 仍大，优先增加 `vc_yaw_rate_abs` 或 `track_ang_vel_z_exp`。
- 如果 IsaacLab 中诊断良好但 MuJoCo 仍差，再进入轻量域随机化阶段。

## 2026-06-01 PPO 标准差负值崩溃修复

### 问题现象
- 训练在 PPO update 阶段报错：`RuntimeError: normal expects all elements of std >= 0.0`。
- 报错位置在 RSL-RL 的 `ActorCritic.act()` 中采样 Normal 分布，说明策略分布的标准差参数被更新成了负值。

### 原因判断
- 当前 RSL-RL 默认 `noise_std_type="scalar"`，会直接学习标准差 `std` 本身；如果 PPO 更新过猛，`std` 可能被梯度推到小于 0。
- 新增运动约束奖励后，reward 尺度和梯度分布发生变化，原来的 `learning_rate=1e-3` 和 `max_grad_norm=1.0` 偏激进。

### 修复内容
- 在 `rsl_rl_ppo_cfg.py` 中设置 `noise_std_type="log"`，改为学习 `log_std`，实际标准差通过指数形式得到，避免标准差为负。
- 将 `init_noise_std` 从 `1.0` 降到 `0.8`，减少初期动作随机幅度。
- 将 `learning_rate` 从 `1.0e-3` 降到 `5.0e-4`。
- 将 `max_grad_norm` 从 `1.0` 降到 `0.5`。
- 将 `entropy_coef` 从 `0.01` 降到 `0.005`，避免在已有较大动作噪声时过度鼓励随机探索。

### 后续观察
- 如果仍出现训练不稳定，下一步优先继续降低 `learning_rate` 到 `3.0e-4`，或临时减小新增约束奖励权重。
- 重新训练时建议不要从旧 checkpoint resume，因为旧 checkpoint 里保存的是 scalar std 参数结构；应重新开一个 run。

## 2026-06-02 主从关节约束与航向漂移约束

### 修改动机
- `--manual_command` 后的 IsaacLab 单点测试已能跟踪 `vx=0.2`，但 MuJoCo sim2sim 中 `wz_mae` 和 `vy_mae` 仍然很高，轨迹经常表现为原地乱扭。
- 当前 7 个 yaw 关节都是独立 action，冗余自由度之间可能互相抵消。为避免“关节打架”，先用 reward 方式建立主动关节/跟随关节关系，不直接修改 action space。

### 新增主从关节约束
- 新增 `leader_follower_joint_pos_l2`：将 `yaw1/yaw3/yaw5/yaw7` 视为主动关节，将 `yaw2/yaw4/yaw6` 视为跟随关节，惩罚跟随关节偏离相邻主动关节平均值。
- 新增 `leader_follower_joint_vel_l2`：对同样的主从关系施加速度跟随约束，减少中间关节反向快速摆动。
- 在 `velocity_env_cfg.py` 中加入 `leader_follower_joint_pos_l2`，权重 `-0.06`；加入 `leader_follower_joint_vel_l2`，权重 `-0.008`。

### 新增航向漂移约束
- 新增 `VirtualChassisHeadingDriftPenalty`：每个 episode 记录 reset 后的 Virtual Chassis 初始 heading，惩罚后续 heading 偏离。
- 在 `velocity_env_cfg.py` 中加入 `vc_heading_drift_l2`，权重 `-0.15`。该项用于直接压制 sim2sim 中持续自转导致的路径偏移。

### 快速验证建议
- 第一轮不必跑满 10000 iteration，可先用 `--max_iterations 2000 --num_envs 4096` 做短跑，观察 `diagnostics/train_monitor.csv` 中 `vel_xy_mae_mean`、`act_wz_abs_mean`、`raw_action_abs_mean` 是否比上一版更早下降。
- 每 1000 或 2000 iteration 用 checkpoint 跑一次 `play.py --manual_command --plot`，优先看 `diagnostics_summary.md` 的 `mae_planar` 和 `act_wz_abs_mean`。
- 短跑有效后再用 8192 env 跑 10000 iteration，并最终跑 `sim2sim_eval.py`。

## 2026-06-02 侧向速度与航向稳定性专项调整

### 修改动机
- `2026-06-02_07-58-16` 相比上一版有改善：`planar_mae` 约下降 8.5%，`vx_mae` 约下降 32%，`wz_mae` 约下降 10%。
- 主要短板仍是 `vy_mae` 偏高且略有变差，说明策略更会前后走，但侧向速度跟踪不足。
- 课程学习此前对 x/y 使用同一上限，可能让 `vy` 扩到不必要的大范围，偏离最终评估范围。

### 课程学习修改
- `command_velocity_curriculum` 新增 `max_curriculum_x/max_curriculum_y` 与 `min_curriculum_x/min_curriculum_y` 参数，并保留旧参数兼容。
- 在 `velocity_env_cfg.py` 中设置 `max_curriculum_x=0.25`、`max_curriculum_y=0.12`、`min_curriculum_x=0.05`、`min_curriculum_y=0.03`。
- 目标是让训练范围更贴近评估分布：vx 可以略宽，vy 保持较窄，避免策略把能力浪费在过大的侧向速度上。

### 奖励函数修改
- 新增 `VirtualChassisTrackVelYExp`，对 Virtual Chassis 坐标系下的 y 速度单独做指数跟踪奖励。
- 在 `SnakeVelocityRewardsCfg` 中加入 `track_lin_vel_y_exp`，权重 `1.2`，`std=0.12`，`linear_coef=0.3`。
- 将 `track_ang_vel_z_exp` 权重从 `1.5` 提高到 `1.8`。
- 将 `vc_yaw_rate_abs` 权重从 `-0.2` 加强到 `-0.3`。
- 将 `vc_heading_drift_l2` 权重从 `-0.15` 加强到 `-0.25`。

### 验证重点
- 短跑时重点观察 sim2sim 的 `vy_mae` 是否从约 `0.19` 降低，同时确认 `vx_mae` 不明显回退。
- 如果 `wz_mae` 继续降低但机器人“不敢动”，优先回调 `vc_heading_drift_l2` 到 `-0.18~-0.20`。
- 如果 `vy_mae` 仍无改善，说明 reward 版约束可能接近瓶颈，下一步应考虑低维 CPG/action 参数化。

## 2026-06-03 sim2sim 泛化专项调整

### 本次实验结果分析
- `2026-06-02_14-38-42` 在 IsaacLab 单点 play 中明显更好：`mae_planar=0.0613`，`mae_vx=0.0255`，`mae_vy=0.0159`，`mae_wz=0.0459`，动作幅度也下降到 `action_abs_mean=1.28`。
- 训练诊断同样显示 PhysX/IsaacLab 内部更稳定：末尾 `vel_xy_mae_mean≈0.053`，`act_wz_abs_mean≈0.052`，`raw_action_abs_mean≈1.65`。
- 但 sim2sim 没有同步改善：相对上一轮 `planar_mae` 从约 `0.3109` 变为 `0.3135`，`vx_mae` 改善到约 `0.0694`，但 `vy_mae` 仍约 `0.1987`，`wz_mae` 约 `0.1824`。
- 结论：继续堆 reward 能让 IsaacLab 曲线变好，但 MuJoCo 迁移收益有限，当前瓶颈更偏向 sim2sim gap 和侧向运动机制不稳。

### 本次代码修改
- 启用轻量域随机化，目标是提高策略对 MuJoCo 物理差异的鲁棒性，而不是进一步提高 IsaacLab 单点曲线。
- 随机化摩擦：`static_friction_range=(0.8, 1.2)`，`dynamic_friction_range=(0.8, 1.2)`。
- 随机化质量：`mass_distribution_params=(0.95, 1.05)`。
- 随机化质心：`x/y/z` 均为 `(-0.002, 0.002)`。
- 随机化 yaw 关节执行器刚度/阻尼：`stiffness_distribution_params=(0.9, 1.1)`，`damping_distribution_params=(0.9, 1.1)`。
- 将训练 `lin_vel_y` 范围从 `[-0.12, 0.12]` 收窄到 `[-0.10, 0.10]`。
- 将课程学习 `max_curriculum_y` 从 `0.12` 收窄到 `0.10`，进一步贴近评估范围。
- 将 `vc_heading_drift_l2` 从 `-0.25` 回调到 `-0.18`，避免过强锁定 heading 导致策略在 MuJoCo 中动作过僵。

### 新的优化思路
- 下一轮重点不再只看 IsaacLab `play.py`，而是以 `sim2sim_eval.py` 的 `vy_mae` 和 `wz_mae` 为主指标。
- 如果轻量域随机化能降低 `wz_mae` 或整体 `planar_mae`，再逐步扩大随机化范围；如果导致 IsaacLab 学不动，则先减小 COM 和 actuator 随机化。
- 如果 sim2sim 仍无改善，说明 7 维独立 yaw action + reward 约束接近瓶颈，下一阶段应转向低维 CPG/action 参数化，让策略输出波幅、相位、方向调制等低维步态参数，而不是直接输出每个 yaw 关节目标。

### 验证建议
- 先短跑 `--max_iterations 3000 --num_envs 4096`，确认训练没有因域随机化明显退化。
- 通过 `diagnostics/train_monitor.csv` 确认 `vel_xy_mae_mean` 不长期高于上一轮，`raw_action_abs_mean` 和 `joint_vel_abs_mean` 不明显暴涨。
- 短跑后必须跑 `sim2sim_eval.py`，本轮是否有效以 `planar_mae`、`vy_mae`、`wz_mae` 是否下降为准。
