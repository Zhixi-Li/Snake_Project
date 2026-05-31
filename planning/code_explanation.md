下面按“这两个脚本到底在干什么”来讲。你可以把整个项目理解成：

IsaacLab 仿真环境 + RSL-RL 强化学习算法 + 蛇形机器人速度跟踪任务配置。

你被要求改的几个文件，本质上是在调这个闭环：
观测 obs -> 神经网络策略 policy -> 动作 action -> IsaacLab 仿真一步 -> 计算奖励 reward / 终止 done -> PPO 更新策略

训练完成后，再把策略导出成 policy.pt，放到 MuJoCo 里做 sim2sim 评估。

1. train.py：训练入口

文件：scripts/rsl_rl/train.py

它不是具体写 PPO 公式的地方，而是一个“把 IsaacLab 环境和 RSL-RL 算法接起来”的启动脚本。

主要流程如下。

第一步，解析命令行参数：

python scripts/rsl_rl/train.py \
--task Snake-VelocityTracking-Flat-v0 \
--num_envs 4096 \
--headless

这里重要参数是：

- --task：选择哪个 Gym 环境。本项目是 Snake-VelocityTracking-Flat-v0。
- --num_envs：并行仿真多少个蛇形机器人。
- --headless：无图形界面训练，速度更快。
- --max_iterations：覆盖 PPO 最大训练迭代数。
- --video：训练时录制视频，调试可用，但会拖慢训练。

第二步，启动 Isaac Sim：

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

IsaacLab 基于 Isaac Sim，所以训练前必须先启动 Omniverse/Isaac Sim 应用。即使你是 headless，它也在后台启动物理仿真器。

第三步，加载任务配置。

关键装饰器是：

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):

它会根据 --task Snake-VelocityTracking-Flat-v0 去 Gym 注册表里找配置。

注册位置是：

source/snake_project/snake_project/tasks/manager_based/velocity_tracking/config/Snake_7DOF/__init__.py

里面注册了：

id="Snake-VelocityTracking-Flat-v0"
env_cfg_entry_point="...flat_env_cfg:SnakeVelocityFlatEnvCfg"
rsl_rl_cfg_entry_point="...rsl_rl_ppo_cfg:SnakeVelocityFlatPPORunnerCfg"

也就是说，训练环境配置来自：

source/snake_project/snake_project/tasks/manager_based/velocity_tracking/config/Snake_7DOF/flat_env_cfg.py

PPO 配置来自：

source/snake_project/snake_project/tasks/manager_based/velocity_tracking/config/Snake_7DOF/agents/rsl_rl_ppo_cfg.py

不过 flat_env_cfg.py 只是继承，真正大部分环境逻辑在：

source/snake_project/snake_project/tasks/manager_based/velocity_tracking/velocity_env_cfg.py

第四步，创建环境：

env = gym.make(args_cli.task, cfg=env_cfg, render_mode=...)

这一步会创建 IsaacLab 的 ManagerBasedRLEnv。它内部会根据配置创建：

- 场景 scene：地面、灯光、机器人、contact sensor
- command manager：速度指令
- observation manager：观测量
- action manager：动作解释方式
- reward manager：奖励函数
- termination manager：终止条件
- event manager：reset 和随机化
- curriculum manager：课程学习

也就是说，IsaacLab 不是在 train.py 里手写 step()，而是用配置类自动拼装一个 RL 环境。

第五步，把 IsaacLab 环境包装成 RSL-RL 可用格式：

env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

RSL-RL 需要一个向量化环境接口。num_envs=4096 时，实际上每一步同时跑 4096 条蛇，得到一大批样本。

第六步，创建 PPO Runner：

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

本项目默认是 OnPolicyRunner，也就是 PPO 这类 on-policy 算法。

最后开始训练：

runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

这行之后，RSL-RL 会循环做：

采样若干步 rollout
计算 advantage
用 PPO loss 更新 actor/critic
保存 checkpoint
写日志

训练结果会在：

logs/rsl_rl/snake_velocity_flat_tracking/时间戳/

里面会有：

- model_*.pt：训练 checkpoint
- params/env.yaml：本次环境配置快照
- params/agent.yaml：本次 PPO 配置快照
- tensorboard 日志

2. play.py：测试、导出、可视化入口

文件：scripts/rsl_rl/play.py

play.py 做的事情不是继续训练，而是加载已有 checkpoint，让策略在 IsaacLab 中跑一遍，并导出 JIT/ONNX 策略。

典型命令：

python scripts/rsl_rl/play.py \
--task Snake-VelocityTracking-Flat-Play-v0 \
--checkpoint logs/rsl_rl/.../model_5000.pt \
--video \
--manual_command \
--cmd_vx 0.2 \
--cmd_vy 0.0

主要流程和 train.py 类似，但目的不同。

第一步，也是启动 Isaac Sim。

第二步，加载 Play 环境：

task_name = args_cli.task.split(":")[-1]
train_task_name = task_name.replace("-Play", "")

训练任务是：

Snake-VelocityTracking-Flat-v0

测试任务是：

Snake-VelocityTracking-Flat-Play-v0

Play 环境在 source/snake_project/snake_project/tasks/manager_based/velocity_tracking/config/Snake_7DOF/flat_env_cfg.py 里覆盖了几个参数：

self.scene.num_envs = 1
self.episode_length_s = 10.0
self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
self.observations.policy.enable_corruption = False

也就是测试时只跑一条蛇，关闭观测噪声。

第三步，找到 checkpoint。

如果你手动传：

--checkpoint your_model.pt

它就加载这个。否则会从日志目录里找默认 checkpoint。

第四步，创建环境并包装：

env = gym.make(args_cli.task, cfg=env_cfg, ...)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

第五步，加载策略：

runner.load(resume_path)
policy = runner.get_inference_policy(device=env.unwrapped.device)

这时得到的是推理用 policy：输入 observation，输出 action。

第六步，导出策略：

export_policy_as_jit(..., filename="policy.pt")
export_policy_as_onnx(..., filename="policy.onnx")

这一步非常关键。README 里 MuJoCo sim2sim 要用的是 JIT policy：

logs/rsl_rl/.../exported/policy.pt

不是训练 checkpoint model_*.pt。

第七步，进入仿真循环：

obs = env.get_observations()

while simulation_app.is_running():
    actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    policy_nn.reset(dones)

这个循环就是测试时的 RL 闭环：

拿 obs -> policy 推理 action -> IsaacLab step -> 得到新 obs

如果你开启：

--manual_command --cmd_vx 0.2 --cmd_vy 0.0

脚本会直接覆盖 command manager 里的速度指令：

command_term.vel_command_b[:, 0] = args_cli.cmd_vx
command_term.vel_command_b[:, 1] = args_cli.cmd_vy
command_term.vel_command_b[:, 2] = args_cli.cmd_wz

所以 play.py 很适合单独检查：某个速度指令下，蛇到底有没有按预期运动。

如果加 --plot，它还会保存：

- Virtual Chassis 速度跟踪图
- Virtual Chassis / base_link 轨迹图
- base_link 速度图

3. 仿真器相关的使用要点

IsaacLab 这里负责训练时的高并行物理仿真。几个你需要理解的点：

num_envs 是并行环境数。
4096 不是跑 4096 次训练，而是在同一个 GPU 仿真里同时放 4096 条蛇，每一步收集 4096 份样本。这是 IsaacLab 训练快的关键。

dt 和 decimation 决定控制频率。
在 source/snake_project/snake_project/tasks/manager_based/velocity_tracking/velocity_env_cfg.py 里：

self.sim.dt = 0.005
self.decimation = 4

物理仿真每 0.005s 积分一次，但策略每 4 个物理步输出一次动作，所以 policy 控制周期是：

0.005 * 4 = 0.02s

也就是 50Hz。MuJoCo 评估脚本也按这个周期构造：

policy_dt_train = 0.005 * 4

所以不要随便改 dt 和 decimation，否则训练策略和 sim2sim 评估节奏会不一致。

headless 适合训练。
图形渲染会拖慢训练。训练时用 --headless，测试和录视频时再打开视频或 GUI。

IsaacLab 训练和 MuJoCo 评估不是同一个仿真器。
训练在 IsaacLab/PhysX，最终评估在 MuJoCo。这就是 sim2sim。你训练中太依赖 IsaacLab 的某些物理细节，到了 MuJoCo 可能表现变差。因此 README 让你考虑域随机化。

4. 强化学习算法基本逻辑

本项目用的是 RSL-RL 的 PPO。你不需要重写 PPO，但要知道它依赖什么。

PPO 有两个网络：

- Actor：输入观测，输出动作。
- Critic：输入观测，估计当前状态的价值。

在配置里：

source/snake_project/snake_project/tasks/manager_based/velocity_tracking/config/Snake_7DOF/agents/rsl_rl_ppo_cfg.py

actor_hidden_dims=[512, 256, 128]
critic_hidden_dims=[512, 256, 128]

Actor 学“该怎么动”，Critic 学“这个状态长期来看好不好”。

每次训练迭代大致是：

1. 当前策略控制蛇运动一段时间。
2. 环境根据速度误差、动作平滑、能耗等算 reward。
3. PPO 计算 advantage：这个动作比预期好还是差。
4. 更新 Actor，让好动作概率更高。
5. 更新 Critic，让价值估计更准。

几个 PPO 参数含义：

- num_steps_per_env=24：每个环境采样 24 步再更新一次。
- max_iterations=5000：训练更新多少轮。
- learning_rate=1e-3：学习率。
- entropy_coef=0.01：鼓励探索，太小可能早熟，太大可能动作发散。
- gamma=0.99：长期奖励折扣。
- lam=0.95：GAE advantage 平滑参数。
- clip_param=0.2：PPO 防止策略一次更新太猛。
- desired_kl=0.01：adaptive schedule 用来控制更新幅度。

你主要不是调算法本体，而是调：

让 PPO 看到什么 obs
让 PPO 输出什么 action
什么行为给 reward
训练速度指令怎么采样
物理参数怎么随机
训练多久、网络多大、学习率多少

5. 被要求修改的文件，本质上是在调什么

核心是 source/snake_project/snake_project/tasks/manager_based/velocity_tracking/velocity_env_cfg.py。

里面每个类对应 RL 闭环的一部分。

SnakeVelocityCommandsCfg：给机器人下什么速度指令。
这里定义目标速度范围：

lin_vel_x=(-0.4, 0.4)
lin_vel_y=(-0.2, 0.2)
ang_vel_z=(0.0, 0.0)

最终 MuJoCo 评估只测 vx=-0.2..0.2、vy=-0.1..0.1。所以调 command 范围，本质上是在决定“训练题库”和“考试题库”是否匹配。

SnakeVelocityActionsCfg：神经网络输出怎么变成机器人控制。
当前是 7 个 yaw 关节的位置控制：

scale=0.25
clip=(-1.57, 1.57)

Actor 输出通常是归一化动作，环境把它乘以 scale，变成关节目标角度。调这里，本质是在调“策略能摆多大、动作是否太激烈”。

SnakeVelocityObservationsCfg：策略能看到什么。
当前 policy 看到：

- base angular velocity
- projected gravity
- velocity command
- yaw joint positions
- yaw joint velocities
- last actions

这决定了 Actor 的输入。注意：Policy 观测必须和 MuJoCo sim2sim 脚本里的观测一致，否则导出的 policy.pt 到 MuJoCo 里输入维度或语义就错了。Critic 可以看到更多 privileged 信息，但
Actor 不行。

SnakeVelocityRewardsCfg：什么行为算好。
当前主要奖励是 Virtual Chassis 速度跟踪：

track_lin_vel_xy_exp
track_ang_vel_z_exp

还有惩罚/辅助项：

- ang_vel_xy_l2：不要乱滚转/俯仰。
- joint_torques_l2：少用过大力矩。
- joint_acc_l2：动作更平滑。
- raw_action_rate：连续动作不要突变。
- joint_amplitude：鼓励关节有运动幅度。
- phase_propagation：鼓励相邻关节形成传播相位。
- motion_coordination：避免所有关节同向乱弯。

调 reward 的本质是：告诉 PPO “什么样的蛇形步态既能跟踪速度，又能平滑稳定，还能迁移到 MuJoCo”。

SnakeVelocityEventCfg：reset 和域随机化。
当前 reset 基本固定，没有开启随机化。README 提示你可以加域随机化。本质是让训练时物理参数有变化，让策略不要只适应 IsaacLab 的一种精确物理条件，从而提高 MuJoCo sim2sim 成功率。

SnakeVelocityCurriculumCfg：课程学习。
课程学习就是不要一开始给策略太难的速度指令。先从小速度学会，再逐步扩大范围。对蛇形机器人这种步态学习任务很有帮助。

rsl_rl_ppo_cfg.py：训练算法参数。
这里调的是训练过程本身，例如训练轮数、网络大小、学习率、探索强度。当前 max_iterations=5000，README 示例是 20000，所以如果效果不好，训练不够也是很可能的原因。

一句话总结

你要完成的不是“写一个新算法”，而是做一个强化学习训练调参任务：

让蛇在 IsaacLab 里学出稳定步态，
让 Actor 的输入/输出和 MuJoCo 评估完全匹配，
让 reward 更贴近最终 MAE 指标，
通过课程学习和轻量域随机化提升 sim2sim 泛化，
最后选出 MuJoCo 25 组速度指令下 MAE 最小的 policy.pt。

建议你之后调试时按这个顺序来：先确认 play.py --manual_command --plot 单速度能跑起来，再跑 sim2sim_eval.py 看 25 组误差分布，然后只改一个因素重新训练比较。这样不会陷入“改了很多
但不知道哪个有效”的状态。