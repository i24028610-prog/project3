"""
（Task 1：真实俄罗斯方块.exe 图像采集）

本项目用于强化学习俄罗斯方块任务1（真实图像检测与数据采集）。
目标是从真实运行的俄罗斯方块.exe程序中采集可用于后续训练的数据集：每一帧输出固定大小18×14的棋盘网格（CSV可读），并同步记录每一帧对应的动作（CSV），保证帧与动作严格对齐。
程序通过屏幕抓取获得游戏画面，首次运行或按下R键时，会在全屏虚拟桌面截图中手动框选棋盘ROI（建议只框白色棋盘内部，避免外框或右侧UI干扰），
ROI会自动保存到out_task1_real/roi_config.json，后续运行会直接读取该配置并开始采集。采集时程序以固定帧率（默认15FPS）循环截取ROI，将每帧棋盘转换为18×14网格并写入frames_YYYYMMDD_HHMMSS.csv，
同时将键盘输入动作按“每帧一条记录”的方式写入actions_YYYYMMDD_HHMMSS.csv，frame_idx从0开始连续递增，确保与frames中的frame编号完全一致。

由于该exe游戏方块落地后会变为灰色，颜色阈值方法不稳定，因此本程序采用边缘能量（Sobel边缘幅值）判断格子是否被占用：0表示空格，1表示占用，
能够同时识别彩色下落块与落地后的灰色块。运行时会显示GUI窗口，仅用于实时检查ROI与识别结果：窗口显示ROI画面、绘制18×14网格线，并在左上角显示当前边缘阈值edge_th与本帧占用格数量filled；
本版本界面更简洁，不影响数据采集与保存。快捷键：按Q退出并自动保存CSV；按R重新选择ROI并覆盖roi_config.json；按C在当前ROI上进行一次阈值校准（建议在开局棋盘较空时按一次，以提高稳定性）。
控制台出现“Select a ROI and then press SPACE or ENTER button!”属于OpenCV ROI选择提示语，并非报错；若出现网格全0或全1等异常，通常是ROI框选不准确或未校准阈值导致，可按R重新框选并在空棋盘时按C重新校准。

依赖环境为Python 3.9+，需要安装opencv-python、numpy、mss、pynput。
安装命令：pip install opencv-python numpy mss pynput。运行方式：python task1.py。
运行结束后，输出文件位于out_task1_real/目录下，包括roi_config.json、frames_*.csv与actions_*.csv，其中frames与actions的行数/帧数应一致且frame编号连续，用于后续任务直接读取训练。
"""
