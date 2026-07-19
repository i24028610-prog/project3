# Task8 程序整理说明

本文档用于说明俄罗斯方块神经网络 Task8 相关程序的作用、运行顺序、二消三消奖励机制修改位置、`.pt` 模型训练入口和结果验证入口。

## 当前任务目标

当前 Task8 的目标已经更新为：

```text
总消行达到 11000 行时结束验证。
消 2 行、消 3 行贡献的行数 / 总消行数 >= 15%。
消 1 行 = 1 分，消 2 行 = 10 分，消 3 行 = 100 分。
```

注意，占比不是按次数算，而是按贡献行数算：

```text
2/3 行贡献占比 =
(2 * 消2行次数 + 3 * 消3行次数) / 总消行数
```

例如：

```text
消1行次数 = 1867
消2行次数 = 378
消3行次数 = 43

总消行数 = 1867 + 2*378 + 3*43 = 2752
2/3行贡献 = 2*378 + 3*43 = 885
2/3行贡献占比 = 885 / 2752 = 32.16%
```

## 文件总览

```text
task8_trajectory_select_v6.py
    轨迹筛选程序：从 selfplay 数据中筛选精英训练数据。

task8_selfimit_train_v6.py
    Task8 主训练程序：训练神经网络并输出 .pt 模型。

task7_task3_sim.py
    模型验证和可视化程序：加载 .pt，运行 Pygame，查看得分、消行、二消三消占比。

task8_v4.py
    早期实验程序：包含 collect / train / eval / search teacher，当前作为历史版本保留。
```

## 1. task8_trajectory_select_v6.py

路径：

```text
C:\Users\25361\PycharmProjects\pythonProject3\task8_trajectory_select_v6.py
```

作用：

这个程序负责“准备训练集”，不直接训练 `.pt` 模型。

它会从自我对局数据中筛选更适合训练的精英样本，包括：

- 存活时间更长的轨迹；
- 消行更多的轨迹；
- 棋盘结构更稳定的轨迹；
- 修复样本；
- 救援样本；
- 稳定样本；
- 健康消行样本；
- 后期 sustain / tail 样本。

默认输入：

```text
C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfplay\selfplay_dataset_v1.npz
C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfplay\episode_metrics_v1.csv
```

默认输出：

```text
out_task8_elite_v6\v6_struct_mainline_v2
```

运行命令：

```powershell
cd C:\Users\25361\PycharmProjects\pythonProject3
.\.venv\Scripts\python.exe task8_trajectory_select_v6.py
```

指定输出标签运行：

```powershell
.\.venv\Scripts\python.exe task8_trajectory_select_v6.py --out_dir out_task8_elite_v6 --tag v6_struct_mainline_v2
```

这个程序什么时候用：

```text
当你有新的 selfplay 数据，需要重新筛选训练集时运行。
如果只是用已有数据继续训练，可以不运行它。
```

## 2. task8_selfimit_train_v6.py

路径：

```text
C:\Users\25361\PycharmProjects\pythonProject3\task8_selfimit_train_v6.py
```

作用：

这是当前 Task8 的主训练程序，负责训练神经网络并生成 `.pt` 模型。

它会：

- 读取 Task8 精英轨迹数据；
- 读取 Task6 基础数据作为稳定器；
- 加载已有初始模型继续训练；
- 使用 legal mask 训练合法动作；
- 加强消 2 行、消 3 行样本的权重；
- 在训练过程中做严格模拟评估；
- 输出 `model_best_action.pt` 和 `model_last_action.pt`。

默认输入数据：

```text
Task8 精英数据：
C:\Users\25361\PycharmProjects\pythonProject3\out_task8_elite_v6\v6_struct_mainline

Task6 基础数据：
C:\Users\25361\PycharmProjects\pythonProject3\out_task6_data

初始模型：
C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_next\model_best_action.pt
```

默认输出目录：

```text
out_task8_selfimit_v6_mainline_v2
```

默认输出模型：

```text
out_task8_selfimit_v6_mainline_v2\model_best_action.pt
out_task8_selfimit_v6_mainline_v2\model_last_action.pt
```

运行训练命令：

```powershell
cd C:\Users\25361\PycharmProjects\pythonProject3
.\.venv\Scripts\python.exe task8_selfimit_train_v6.py
```

推荐新目标训练命令：

```powershell
.\.venv\Scripts\python.exe task8_selfimit_train_v6.py --out_dir out_task8_selfimit_v6_combo_11000_ratio15
```

从当前稳定模型继续训练：

```powershell
.\.venv\Scripts\python.exe task8_selfimit_train_v6.py `
  --init_model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt" `
  --out_dir out_task8_selfimit_v6_combo_11000_ratio15
```

快速短训测试命令：

```powershell
.\.venv\Scripts\python.exe task8_selfimit_train_v6.py `
  --init_model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt" `
  --out_dir out_task8_selfimit_v6_combo_11000_ratio15_fast `
  --epochs 3 `
  --heldout_eval_episodes 8 `
  --shadow_eval_episodes 8 `
  --heldout_eval_max_pieces 500 `
  --shadow_eval_max_pieces 500
```

重要参数：

```text
DEFAULT_EVAL_POLICY_TOPK = 56
DEFAULT_EVAL_SAFETY_WEIGHT = 1.20
DEFAULT_EVAL_COMBO_WEIGHT = 0.30
TARGET_COMBO_CLEAR_RATIO = 0.15
TARGET_TOTAL_LINES = 11000
```

这个程序什么时候用：

```text
需要重新训练 .pt 模型时使用。
它是 Task8 当前最重要的训练入口。
```

## 3. task7_task3_sim.py

路径：

```text
C:\Users\25361\PycharmProjects\pythonProject3\task7_task3_sim.py
```

作用：

这是当前推荐的模型运行和结果查看程序。

它负责：

- 加载训练好的 `.pt` 模型；
- 在 Pygame 界面中运行俄罗斯方块自动玩家；
- 显示总消行、任务得分、已落块数量；
- 显示消 1 行、消 2 行、消 3 行各发生几次；
- 显示 2/3 行贡献占比；
- 显示当前是否达到 `>= 15%`；
- 当总消行达到 11000 行时自动结束验证。

当前推荐模型：

```text
C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt
```

运行验证命令：

```powershell
cd C:\Users\25361\PycharmProjects\pythonProject3
.\.venv\Scripts\python.exe task7_task3_sim.py --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt"
```

固定 seed 验证命令：

```powershell
.\.venv\Scripts\python.exe task7_task3_sim.py `
  --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt" `
  --fixed_seed `
  --seed 9101
```

显式指定策略参数：

```powershell
.\.venv\Scripts\python.exe task7_task3_sim.py `
  --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt" `
  --policy_topk 56 `
  --safety_weight 1.20 `
  --combo_weight 0.30
```

验证新训练模型：

```powershell
.\.venv\Scripts\python.exe task7_task3_sim.py `
  --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_combo_11000_ratio15\model_best_action.pt" `
  --policy_topk 56 `
  --safety_weight 1.20 `
  --combo_weight 0.30
```

这个程序什么时候用：

```text
训练好 .pt 以后，用它运行并直观看结果。
它不是训练程序，是验证和可视化程序。
```

## 4. task8_v4.py

路径：

```text
C:\Users\25361\PycharmProjects\pythonProject3\task8_v4.py
```

作用：

这是早期 Task8 实验程序，当前不作为主线入口。

它包含：

- `collect`：采集早期训练数据；
- `collect_relabel`：用 student 模型重新标注数据；
- `train`：训练早期版本模型；
- `eval`：评估早期模型；
- `eval_search`：评估搜索教师策略。

查看帮助：

```powershell
cd C:\Users\25361\PycharmProjects\pythonProject3
.\.venv\Scripts\python.exe task8_v4.py --help
```

采集数据：

```powershell
.\.venv\Scripts\python.exe task8_v4.py --mode collect --out_dir out_task8_stage2_model
```

训练旧版模型：

```powershell
.\.venv\Scripts\python.exe task8_v4.py --mode train --out_dir out_task8_stage2_model
```

评估旧版模型：

```powershell
.\.venv\Scripts\python.exe task8_v4.py --mode eval --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_stage2_model\model_best_action.pt"
```

这个程序什么时候用：

```text
只在需要回看旧实验、旧搜索教师、旧数据采集流程时使用。
当前正式 Task8 训练不推荐从这里继续改。
```

## 二消三消奖励机制改在哪里

### A. 验证程序里的修改位置

文件：

```text
task7_task3_sim.py
```

主要修改点：

```text
TARGET_COMBO_CLEAR_RATIO = 0.15
TARGET_TOTAL_LINES = 11000
DEFAULT_COMBO_WEIGHT = 0.30
combo_clear_ratio(...)
combo_setup_score(...)
combo_action_value(...)
NNPolicy.choose_safe_action(...)
TetrisGame.lock_piece(...)
PolishedTetrisRenderer.draw_side_panel(...)
```

具体作用：

- `combo_clear_ratio(...)`  
  计算 2/3 行贡献占比：

```text
(2 * 消2行次数 + 3 * 消3行次数) / 总消行数
```

- `combo_setup_score(...)`  
  奖励能制造后续二消、三消机会的棋盘形状。

- `combo_action_value(...)`  
  对当前动作打组合消行分：

```text
消3行：最高奖励
消2行：较高奖励
消1行：降低吸引力
不消行：如果能制造二消三消机会，也给布局奖励
```

- `NNPolicy.choose_safe_action(...)`  
  在神经网络给出的合法动作中重新排序，综合：

```text
神经网络原始分
安全分
二消三消奖励分
是否危险局面
是否灾难动作
```

局面策略：

```text
健康局面：更主动追求二消三消
普通局面：适当提高二消三消权重
危险局面：先保命，只保留更安全的候选动作
```

- `TetrisGame.lock_piece(...)`  
  每次锁定方块后统计：

```text
总消行数
消1行次数
消2行次数
消3行次数
任务得分
是否达到 11000 总消行结束条件
```

- `PolishedTetrisRenderer.draw_side_panel(...)`  
  在界面上显示：

```text
总消行 / 11000
消1行次数
消2行次数
消3行次数
2/3行贡献占比
达标 / 未达标
```

### B. 训练程序里的修改位置

文件：

```text
task8_selfimit_train_v6.py
```

主要修改点：

```text
TARGET_COMBO_CLEAR_RATIO = 0.15
TARGET_TOTAL_LINES = 11000
DEFAULT_EVAL_COMBO_WEIGHT = 0.30
combo_clear_ratio(...)
multi_clear_bonus
combo_setup_score_eval(...)
combo_action_value_eval(...)
choose_action_greedy(...)
run_eval(...)
dyn_weight
```

具体作用：

- `multi_clear_bonus`  
  在数据集样本权重中提高二消、三消样本的重要性：

```text
消1行样本：+0.10
消2行样本：+3.00
消3行及以上样本：+8.00
```

- `dyn_weight`  
  训练 batch 内再次提高二消、三消样本权重：

```text
lines_gain == 1：+0.10
lines_gain == 2：+3.00
lines_gain >= 3：+8.00
```

- `combo_action_value_eval(...)`  
  训练评估阶段也使用和验证程序一致的二消三消奖励逻辑。

- `choose_action_greedy(...)`  
  训练评估时不再只看神经网络 argmax，而是用：

```text
policy score + safety score + combo score
```

- `run_eval(...)`  
  训练过程中输出：

```text
mean_combo_clear_ratio
heldout_mean_combo_clear_ratio
shadow_mean_combo_clear_ratio
```

并且模型选择指标会：

```text
低于 15%：强惩罚
高于 15%：继续奖励
```

## 当前推荐完整流程

### 第一步：如果有新的 selfplay 数据，先筛选轨迹

```powershell
cd C:\Users\25361\PycharmProjects\pythonProject3
.\.venv\Scripts\python.exe task8_trajectory_select_v6.py --out_dir out_task8_elite_v6 --tag v6_struct_mainline_v2
```

如果没有新 selfplay 数据，可以跳过这一步。

### 第二步：训练 Task8 模型并生成 pt

```powershell
cd C:\Users\25361\PycharmProjects\pythonProject3
.\.venv\Scripts\python.exe task8_selfimit_train_v6.py `
  --init_model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt" `
  --out_dir out_task8_selfimit_v6_combo_11000_ratio15
```

训练完成后重点看：

```text
out_task8_selfimit_v6_combo_11000_ratio15\model_best_action.pt
```

### 第三步：运行验证程序查看结果

```powershell
cd C:\Users\25361\PycharmProjects\pythonProject3
.\.venv\Scripts\python.exe task7_task3_sim.py `
  --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_combo_11000_ratio15\model_best_action.pt" `
  --policy_topk 56 `
  --safety_weight 1.20 `
  --combo_weight 0.30
```

如果先验证当前稳定模型：

```powershell
.\.venv\Scripts\python.exe task7_task3_sim.py `
  --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt" `
  --policy_topk 56 `
  --safety_weight 1.20 `
  --combo_weight 0.30
```

### 第四步：固定 seed 做可复现实验

```powershell
.\.venv\Scripts\python.exe task7_task3_sim.py `
  --model "C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt" `
  --policy_topk 56 `
  --safety_weight 1.20 `
  --combo_weight 0.30 `
  --fixed_seed `
  --seed 9101
```

## 最终检查标准

在验证界面中重点看：

```text
总消行：达到 11000 / 11000
消2行次数：越高越好
消3行次数：越高越好
2/3行贡献占比：必须 >= 15%
状态：达成目标
```

如果界面显示：

```text
2/3行贡献占比 >= 15%    达标
```

说明当前模型满足二消三消占比要求。

如果显示：

```text
未达标
```

说明需要继续提高 `combo_weight` 或继续训练模型。

## 哪个程序训练 pt，哪个程序看结果

```text
训练 pt：
task8_selfimit_train_v6.py

查看结果：
task7_task3_sim.py

筛选训练数据：
task8_trajectory_select_v6.py

旧实验备用：
task8_v4.py
```

