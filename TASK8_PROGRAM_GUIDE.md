# Task8 程序整理说明

本文档用于说明当前俄罗斯方块神经网络 Task8 相关程序的分工、作用、运行顺序和推荐使用的模型文件。

## 当前目标

当前 Task8 的优先级已经调整为：

1. 首先保证长期存活，不能轻易死亡。
2. 在保证安全和长期存活的前提下，尽量消 2 行、消 3 行。
3. 新计分规则为：消 1 行 = 1 分，消 2 行 = 10 分，消 3 行 = 100 分。

因此当前策略不是盲目追求三消，而是：

```text
先保命 -> 再在安全候选动作中追求 2 行 / 3 行消除
```

## 核心文件说明

### 1. task8_selfimit_train_v6.py

路径：

```text
C:\Users\25361\PycharmProjects\pythonProject3\task8_selfimit_train_v6.py
```

作用：

这是当前 Task8 的主训练程序。

它负责：

- 加载 Task8 精英轨迹数据；
- 加载 Task6 基础数据作为稳定器；
- 继续训练神经网络策略模型；
- 保存新的 `model_best_action.pt` 和 `model_last_action.pt`；
- 在训练过程中用严格模拟环境评估模型；
- 使用 `policy_topk` 和 `safety_weight` 做生存优先的动作重排。

当前重要策略：

```text
CLEAR_SCORE_TABLE = [0, 1, 10, 100, 100]
SURVIVAL_CLEAR_SCORE_WEIGHT = 0.05
eval_policy_topk = 56
eval_safety_weight = 1.20
```

说明：

- `CLEAR_SCORE_TABLE` 表示新任务的消行得分。
- `SURVIVAL_CLEAR_SCORE_WEIGHT = 0.05` 表示消 2/3 行只是安全前提下的加分，不会压过生存安全。
- `eval_policy_topk = 56` 表示尽量检查全部合法动作，而不是只看神经网络最高分。
- `eval_safety_weight = 1.20` 表示更重视棋盘结构安全。

推荐运行训练：

```powershell
cd C:\Users\25361\PycharmProjects\pythonProject3
.\.venv\Scripts\python.exe task8_selfimit_train_v6.py
```

如果要重新训练当前“生存优先 + 安全二消三消”版本，可以指定输出目录：

```powershell
.\.venv\Scripts\python.exe task8_selfimit_train_v6.py --out_dir out_task8_selfimit_v6_survival_first
```

### 2. task7_task3_sim.py

路径：

```text
C:\Users\25361\PycharmProjects\pythonProject3\task7_task3_sim.py
```

作用：

这是当前推荐的模型验证和可视化程序。

它负责：

- 加载训练好的 `.pt` 模型；
- 在 Pygame 里运行俄罗斯方块自动玩家；
- 用神经网络预测合法动作；
- 对全部合法动作进行生存优先筛选；
- 显示消行数、组合得分、1/2/3 行消除次数；
- 验证模型是否能够长期存活。

当前推荐验证模型：

```text
C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt
```

推荐验证命令：

```powershell
cd C:\Users\25361\PycharmProjects\pythonProject3
.\.venv\Scripts\python.exe task7_task3_sim.py --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt" --policy_topk 56 --safety_weight 1.20
```

固定种子验证命令：

```powershell
.\.venv\Scripts\python.exe task7_task3_sim.py --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt" --policy_topk 56 --safety_weight 1.20 --fixed_seed --seed 9101
```

说明：

- 这个程序不是单纯 `argmax`。
- 它会先取合法候选动作，再模拟落子后的棋盘结构。
- 如果存在安全候选，只在安全候选里选择最优动作。
- 如果所有候选都危险，选择相对最不容易死亡的动作。

### 3. task8_trajectory_select_v6.py

路径：

```text
C:\Users\25361\PycharmProjects\pythonProject3\task8_trajectory_select_v6.py
```

作用：

这是 Task8 的轨迹筛选程序。

它负责：

- 从自我对局数据中筛选精英样本；
- 按存活时间、消行、结构稳定性筛选轨迹；
- 标记修复样本、救援样本、稳定样本、健康消行样本；
- 生成训练用的精英数据目录。

输入数据通常来自：

```text
out_task8_selfplay
```

输出数据通常到：

```text
out_task8_elite_v6
```

当前主训练程序默认使用的精英数据目录：

```text
C:\Users\25361\PycharmProjects\pythonProject3\out_task8_elite_v6\v6_struct_mainline
```

### 4. task8_v4.py

路径：

```text
C:\Users\25361\PycharmProjects\pythonProject3\task8_v4.py
```

作用：

这是早期 Task8 实验程序。

它包含：

- 搜索教师策略；
- 采集训练数据；
- relabel 数据；
- 早期训练和评估逻辑。

当前状态：

```text
保留作为历史实验版本，不作为当前推荐入口。
```

当前推荐不要优先运行它，除非需要重新生成早期搜索教师数据。

## 当前推荐模型文件

### 首选稳定模型

```text
C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt
```

用途：

- 当前最推荐用于“不挂、长期存活”的验证；
- 已通过 10 局、每局 1000 块上限压力测试；
- 测试中死亡局数为 0。

### 新目标短训模型

```text
C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_combo_survival_fast\model_best_action.pt
```

用途：

- 用新计分目标进行过短轮数训练；
- 可作为后续继续训练的起点；
- 目前仍建议先用 mainline_v2 的 best 模型验证生存稳定性。

## 当前推荐工作流

### 第一步：验证不挂

运行：

```powershell
.\.venv\Scripts\python.exe task7_task3_sim.py --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt" --policy_topk 56 --safety_weight 1.20
```

目标：

```text
先保证不会轻易死亡。
```

### 第二步：在不挂基础上优化二消三消

运行训练：

```powershell
.\.venv\Scripts\python.exe task8_selfimit_train_v6.py --out_dir out_task8_selfimit_v6_survival_first
```

目标：

```text
保持长期存活，同时逐步增加 2 行 / 3 行消除。
```

### 第三步：用验证程序测试新 pt

示例：

```powershell
.\.venv\Scripts\python.exe task7_task3_sim.py --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_survival_first\model_best_action.pt" --policy_topk 56 --safety_weight 1.20
```

## 文件关系总结

```text
task8_v4.py
    早期搜索教师 / 数据采集 / 实验版本

task8_trajectory_select_v6.py
    从 selfplay 数据中筛选精英轨迹

task8_selfimit_train_v6.py
    当前 Task8 主训练程序，生成新的 pt 模型

task7_task3_sim.py
    当前推荐验证程序，加载 pt 并进行 Pygame 可视化验证
```

## 当前结论

当前阶段的基础目标是“不挂”。因此推荐组合是：

```text
验证程序：
task7_task3_sim.py

模型文件：
out_task8_selfimit_v6_mainline_v2\model_best_action.pt

策略参数：
--policy_topk 56 --safety_weight 1.20
```

在这个组合下，当前压力测试结果为：

```text
10 局固定种子测试
每局 1000 块上限
死亡局数 0
全部达到 1000 块
```
