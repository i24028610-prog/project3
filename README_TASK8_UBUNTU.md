# Task8 Ubuntu 真机运行说明

本文档用于在 Ubuntu 双系统中运行 Task8 俄罗斯方块神经网络程序。要求是在真实 Ubuntu 系统中运行，不使用虚拟机。

## 1. Ubuntu 环境要求

建议使用：

```text
Ubuntu 22.04 / 24.04
Python 3.10 或 3.11
真实桌面环境
```

安装系统依赖：

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git fonts-noto-cjk fonts-wqy-microhei
```

如果要用 NVIDIA GPU 训练，需要先在 Ubuntu 中安装好显卡驱动和对应 CUDA 版本。只做 Pygame 验证时可以直接使用 CPU。

## 2. 获取项目

推荐在 Ubuntu 中 clone GitHub 仓库：

```bash
cd ~
git clone https://github.com/i24028610-prog/project3.git
cd project3
```

如果使用 Windows 分区里的同一份项目，也可以进入挂载目录，例如：

```bash
cd /mnt/c/Users/25361/PycharmProjects/pythonProject3
```

更推荐 clone 到 Ubuntu 自己的 home 目录，因为训练时读写大量 `.npz`、`.csv`、`.pt` 文件会更稳定。

## 3. 创建 Python 虚拟环境

```bash
cd ~/project3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_ubuntu.txt
```

如果 GPU 版 PyTorch 需要单独安装，请按你 CUDA 版本安装对应的 `torch`。安装完成后可以检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

## 4. Ubuntu 可运行适配点

当前代码已经做了跨平台处理：

```text
task7_task3_sim.py
    默认模型路径改为相对项目目录。
    中文字体同时支持 Windows 字体和 Ubuntu Noto/WenQuanYi 字体。

task8_selfimit_train_v6.py
    默认 elite_data_dir、base_data_dir、init_model 改为相对项目目录。

task8_trajectory_select_v6.py
    默认 selfplay_npz、episode_csv 改为相对项目目录。
```

所以在 Ubuntu 中不再需要把代码里的 `C:\Users\...` 改成 Linux 路径。

## 5. 验证已有 pt 模型

当前推荐验证程序：

```text
task7_task3_sim.py
```

当前推荐模型：

```text
out_task8_selfimit_v6_mainline_v2/model_best_action.pt
```

运行：

```bash
cd ~/project3
source .venv/bin/activate
python task7_task3_sim.py \
  --model out_task8_selfimit_v6_mainline_v2/model_best_action.pt \
  --policy_topk 56 \
  --safety_weight 1.20 \
  --combo_weight 0.30
```

固定 seed 复现实验：

```bash
python task7_task3_sim.py \
  --model out_task8_selfimit_v6_mainline_v2/model_best_action.pt \
  --policy_topk 56 \
  --safety_weight 1.20 \
  --combo_weight 0.30 \
  --fixed_seed \
  --seed 9101
```

界面中重点看：

```text
总消行是否达到 11000 / 11000
消2行次数
消3行次数
2/3行贡献占比是否 >= 15%
状态是否达成目标
```

## 6. 重新筛选训练数据

筛选程序：

```text
task8_trajectory_select_v6.py
```

运行：

```bash
python task8_trajectory_select_v6.py \
  --out_dir out_task8_elite_v6 \
  --tag v6_struct_mainline_v2
```

它默认读取：

```text
out_task8_selfplay/selfplay_dataset_v1.npz
out_task8_selfplay/episode_metrics_v1.csv
```

输出给训练程序使用的精英数据目录。

## 7. 重新训练 pt 模型

训练程序：

```text
task8_selfimit_train_v6.py
```

从当前稳定模型继续训练：

```bash
python task8_selfimit_train_v6.py \
  --init_model out_task8_selfimit_v6_mainline_v2/model_best_action.pt \
  --elite_data_dir out_task8_elite_v6/v6_struct_mainline \
  --base_data_dir out_task6_data \
  --out_dir out_task8_selfimit_v6_combo_11000_ratio15 \
  --eval_policy_topk 56 \
  --eval_safety_weight 1.20 \
  --eval_combo_weight 0.30
```

训练完成后重点使用：

```text
out_task8_selfimit_v6_combo_11000_ratio15/model_best_action.pt
```

然后用 `task7_task3_sim.py` 加载这个新 `.pt` 验证。

## 8. 每个程序做什么

```text
task8_trajectory_select_v6.py
    筛选训练轨迹，不训练 pt。

task8_selfimit_train_v6.py
    正式训练 Task8 的 pt 模型。

task7_task3_sim.py
    加载 pt 并运行可视化验证。

task8_v4.py
    早期实验程序，当前不作为主线。

TASK8_PROGRAM_GUIDE.md
    Windows / 总体任务说明。

README_TASK8_UBUNTU.md
    Ubuntu 真机运行说明。
```

## 9. 常见问题

如果 Pygame 窗口无法打开，先确认你是在 Ubuntu 桌面环境中运行：

```bash
echo $DISPLAY
```

如果中文显示成方块，重新安装字体：

```bash
sudo apt install -y fonts-noto-cjk fonts-wqy-microhei
```

如果提示找不到模型文件，确认 `.pt` 文件存在：

```bash
ls -lh out_task8_selfimit_v6_mainline_v2/model_best_action.pt
```

如果提示 CUDA 不可用，可以先加 `--cpu` 验证：

```bash
python task7_task3_sim.py \
  --model out_task8_selfimit_v6_mainline_v2/model_best_action.pt \
  --cpu
```
