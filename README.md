# dark-image-detection
# 项目名称
基于 YOLO 的低光目标检测

## 项目简介
本项目用于在 ExDark 数据集上进行低光目标检测实验，当前包含：
- YOLO 基线检测
- 轻量级特征增强模块 `feat_enhance_lite`
- 检测与增强联合训练

## 主要目录
- train.py：训练入口
- test.py：验证入口
- models/：YOLO 网络结构与增强模块
- models/feat_enhance_lite.py：增强模块
- data/：数据集配置
- cfg/：模型配置
- utils/：训练与推理辅助函数

## 环境
- Python 3.8 
- PyTorch 1.12
- CUDA 11.6 或按服务器环境配置

## 常用命令
训练：
python train.py --data data/exdark.yaml --cfg cfg/training/yolov7.yaml --weights yolov7.pt --epochs 50 --batch-size 6

验证：
python test.py --data data/exdark.yaml --weights runs/train/exp/weights/best.pt

## 数据集说明
使用 ExDark 数据集，采用 YOLO 格式标注。

## 当前实验说明
当前增强模块集成在 `models` 目录下，模块名称为 `feat_enhance_lite`。
该模块用于低光场景下的特征增强，可与 YOLO 检测网络进行联合训练。
