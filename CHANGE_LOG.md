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
