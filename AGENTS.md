# AGENTS.md

## 项目目标
这是一个用于低光目标检测实验的 YOLO 项目，重点研究：
- 弱光图像增强
- 检测与增强联合训练
- 在 ExDark 数据集上的性能提升

## 工作原则
- 优先做最小改动，避免无关重构
- 保持原有训练流程可运行
- 新增模块时，尽量不要破坏 baseline
- 修改前先阅读相关文件，再动手
- 修改后给出改动说明

## 代码规范
- 尽量保持现有代码风格
- 不随意改变量名和目录结构
- 增加必要注释，但不要过度冗长
- 不删除已有功能，除非明确说明原因

## 重点文件
- train.py：训练主入口
- models/yolo.py：模型结构
- data/exdark.yaml：数据配置
- models/feat_enhancer_lite.py：增强模块
- test.py：验证脚本

## 修改限制
- 不要改动数据集真实路径，除非明确要求
- 不要删除已有实验配置文件
- 不要自动改动 runs/ 下的实验结果
- 不要假设用户本地一定有完整数据集

## 验收要求
每次改动后，优先确保：
1. 代码能正常 import
2. 训练命令参数不报错
3. forward 维度逻辑正确
4. 联合损失能正常回传
5. 不影响原始 baseline 的独立运行

## 常用命令
训练示例：
python train.py --data data/exdark.yaml --cfg cfg/training/yolov7.yaml --weights yolov7.pt --batch-size 6 --epochs 30

继续训练示例：
python train.py --data data/exdark.yaml --cfg cfg/training/yolov7.yaml --weights runs/train/exp/weights/last.pt --batch-size 6 --epochs 50

验证示例：
python test.py --data data/exdark.yaml --weights runs/train/exp/weights/best.pt

## 当用户要求修改代码时
请先：
1. 说明你准备改哪些文件
2. 解释改动目的
3. 尽量给出最小补丁
4. 避免引入与需求无关的新模块
