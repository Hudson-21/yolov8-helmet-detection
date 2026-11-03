# 基于 YOLOv8 的安全帽佩戴检测系统

> 行业工程实践 · YOLO 视觉模块课程设计
> 技术栈：YOLOv8n + BoT-SORT 跟踪 + tkinter 可视化界面 + PyTorch

对作业画面中的**头部**做二分类检测（`hardhat` 已戴 / `no-hardhat` 未戴），
检测到未戴目标即触发告警，支持图片、视频、摄像头三种输入，并统计人数与实时 FPS。

| 指标 | 数值 |
|---|---|
| 验证集 mAP@0.5 | **0.9354** |
| 验证集 mAP@0.5:0.95 | 0.5813 |
| Precision / Recall | 0.9030 / 0.8930 |
| 推理速度（纯检测） | 42.9 FPS |
| 视频端到端（含编解码写出） | 19.5 FPS |
| 界面摄像头实测 | 约 60 FPS |

---

## 一、这个项目做了什么

1. **数据集二次加工**：公开数据集 → COCO 转 YOLO → 剔退化框 / 去重 / 统一分辨率 / 8:1:1 重划分 → 全量标注校验。
   最终 19441 张图、55381 个框，校验零错误。
2. **网络结构改进尝试**（如实记录结果）：在 Ultralytics 源码中新增 CBAM / SE 模块并注册进 `parse_model`，
   另新增 P2 小目标检测头（四尺度）。**三组消融均未超过基线**，最终部署原生 YOLOv8n。
   详见 [docs/03_项目文档.md](docs/03_项目文档.md) 与下方消融表。
3. **业务逻辑**：不是简单数框——用 BoT-SORT 的 track id 去重计数，检测到 `no-hardhat` 即告警并带冷却。
4. **工程完整**：数据清洗、划分、校验、训练、评估、推理、界面、导出全链路脚本化，可复现。

## 二、消融实验（为什么"改进"没生效）

三组实验使用同一份数据划分、同一套超参数、相同 25 轮，仅结构不同：

| 实验 | 结构改动 | batch | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| **baseline（最终部署）** | 原生 YOLOv8n | 16 | 0.9030 | 0.8930 | **0.9354** | **0.5813** |
| +CBAM | SPPF 与颈部共 4 处注意力 | 16 | 0.9093 | 0.8856 | 0.9348 | 0.5769 |
| +P2 | 新增 P2 四尺度检测头 | 8 | 0.8910 | 0.8856 | 0.9304 | 0.5730 |

结论：

- 三者 mAP@0.5 差异均在 **0.5 个百分点以内**，属随机波动；
- CBAM 未带来收益，与「数据集仅 2 类、目标特征区分度极高（黄色安全帽 vs 裸露头部）、
  COCO 预训练已提供充分可分性」一致——注意力机制擅长的"抑制复杂背景"在此无用武之地；
- P2 与另两组批大小不同（显存所限），差异不能完全归因于结构，**故不判定为有效改进**。

> 负结果照实保留。能把"改进为什么没用"讲清楚，比虚报提升更有价值。

## 三、目录结构

```
safety_helmet/
├── configs/
│   ├── safety_helmet.yaml      # 数据集配置（nc=2, hardhat / no-hardhat）
│   └── hyp_tuned.yaml          # 超参配置（每条改动带注释理由）
├── src/
│   ├── config.py               # 全局配置、路径、类别、权重解析
│   ├── detect.py               # 检测核心：推理 + 跟踪 + 计数 + 告警 + 绘制
│   ├── train.py                # 训练入口（支持断点续训）
│   ├── val.py                  # 评估（mAP/P/R/FPS，支持多权重对比）
│   ├── predict.py              # 推理（图片/视频/摄像头）
│   ├── ui_app.py               # tkinter 图形界面
│   ├── export.py               # ONNX / TorchScript 导出
│   ├── split_dataset.py        # 清洗去重 + 划分
│   ├── check_labels.py         # 标注全量校验
│   └── collect_images.py       # 摄像头抓拍采集
├── tools/
│   ├── coco2yolo.py            # COCO → YOLO 转换 + 初筛
│   ├── make_arch_diagram.py    # 生成系统架构图
│   └── ...                     # 报告生成等辅助脚本
├── framework_patch/            # 对 Ultralytics 源码的改动（见下）
├── weights/trained/best.pt     # 最终模型
├── docs/                       # 项目文档、架构图、课程报告
├── run_ui.bat                  # 一键启动界面
├── setup.bat                   # 环境安装与自检
└── requirements.txt
```

## 四、快速开始

### 1. 环境

```bash
conda create -n yolov8 python=3.10 -y
conda activate yolov8
pip install -r requirements.txt

# 关键：必须把改过的 ultralytics 源码以可编辑方式安装
# 否则加载自定义模型配置会报 NameError: CBAMBlock is not defined
cd <ultralytics 源码目录>
pip install -e .
```

`framework_patch/` 里是本项目对 Ultralytics 的三处改动，按该目录的 `改动说明.md` 放回源码即可。

### 2. 运行

```bash
# 图形界面（Windows 可直接双击 run_ui.bat）
python src/ui_app.py

# 命令行推理
python src/predict.py --source path/to/image.jpg
python src/predict.py --source path/to/video.mp4
python src/predict.py --source 0            # 摄像头

# 训练
python src/train.py --model baseline --name my_exp --batch 16 --epochs 25

# 断点续训
python src/train.py --resume runs/my_exp/weights/last.pt

# 评估
python src/val.py --weights weights/trained/best.pt --split test
```

### 3. 模型导出（ONNX）

仓库不附带导出的 ONNX（12 MB，受提交网络限制），一条命令即可重导：

```bash
python src/export.py --format onnx          # FP32，12.2 MB
python src/export.py --format onnx --half   # FP16，进一步提速
```

已实测的导出结果（可复现）：

| 验证项 | 结果 |
|---|---|
| `onnx.checker.check_model` | 通过 |
| 输入 / 输出 | `1×3×640×640` → `1×6×8400`（2 类 ×(4 框 + 2 类别)） |
| 与 PyTorch 同输入对比 | 最大绝对误差 **4.88e-4**（等价） |
| onnxruntime | 可正常加载推理 |

### 4. 数据准备

数据集本体未纳入版本库（约 800 MB）。复现方式见 [docs/01_数据采集与标注规范.md](docs/01_数据采集与标注规范.md)，
核心三步：

```bash
python tools/coco2yolo.py --zips <train.zip> <valid.zip> <test.zip> --out datasets/raw/hardhat_all
python src/split_dataset.py --raw datasets/raw/hardhat_all --resize 640 --clean
python src/check_labels.py
```

## 五、类别定义

| id | 名称 | 含义 |
|---|---|---|
| 0 | `hardhat` | 戴着安全帽的头部 |
| 1 | `no-hardhat` | 没有戴安全帽的头部（**告警类别**） |

标注粒度为**头部**而非整个人体——这样"戴没戴"是直接判定的，不需要再做人员框与帽子框的空间关联，
避免了手拎帽子、画面边缘半个身子、旁人凑数这三类误判。

## 六、已知问题与改进方向

- 数据集为公开数据，缺少真实作业现场的逆光、夜间、雨雾、粉尘样本，跨域泛化未验证；
- 小目标占比 60.1%，`conf` 不宜沿用通用默认值 0.25（实测该值下远景头部会全部漏检，0.15 才正常）；
- 告警冷却曾有一处实现缺陷（本帧无违规时把计数清零，导致隔帧重复告警），已改为按帧衰减，
  实测 100 帧触发 4 次、间隔约 31 帧；
- 部署侧仅完成 ONNX 导出验证，未在嵌入式设备实测。

## 七、许可

课程设计作品，仅供学习参考。
