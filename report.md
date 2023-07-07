---
title: sandplay classification
date: 2023-07-04 10:21:56
excerpt: 为啥要做？做完后有何收获感想体会？
tags: #UCAS-homewor #image-classification
rating: ⭐
status: inprogress
destination: 
share: false
obsidianUIMode: source
---
### 目的和预期

<!-- 为什么要做这个实验，要解决什么问题，有什么预期（假设）？-->

- 本次实验的目标是完成对沙盒图像的主题分类。本次实验的数据集共有309张沙盒图像，每张图像提供6个专家主题标注。主题共有6大类：整合、流动、联结、分裂、混乱、空洞。数据集划分为训练集260张图像、测试集49张图像。
- 本实验预期使用深度神经网络模型、预训练模型、传统机器学习模型实现沙盒主题分类并对比不同方法之间的性能差异。

### 方案设计

<!-- 实验组和对照组，控制的变量是什么，观察什么指标，如何验证假设预期？-->

- 实验一：比较不同预训练模型在沙盒主题分类任务中的性能
  - 使用在沙盒图像数据集上预训练的Vit-B/16模型进行视觉特征提取，设计并训练分类网络实现主题分类。
  - 使用预训练的resnet-50进行特征提取和分类
- 实验二：比较使用预训练权重和不适用预训练权重的分类性能
  - 使用预训练的resnet-50进行特征提取和分类
  - 使用resnet-50从头训练实现沙盒主题分类
- 实验三：比较不同神经网络模型的分类性能
  - 使用resnet-50从头训练实现沙盒主题分类
  - 使用自主设计的模型实现沙盒主题分类
- 实验四：比较神经网络模型和机器学习模型的分类性能
  - 使用预训练Vit-B/16进行特征提取并利用KNN实现分类 ，与先前模型对比
- 实验五：比较监督学习和无监督学习的分类性能
  - 使用预训练Vit-B/16进行特征提取并利用K-means实现分类，与先前模型对比

### 实验步骤

<!-- 具体的实验步骤，按顺序一条一条写，或者使用表格。-->

- 统一标签：本次实验用到的数据集每个图像包含6个专家标注，每个专家标注给出了若干个可能的主题类别。因此需要设计方案来统一每张图像的具体类别。
  - 实现的思路是统计6个专家标注，采用投票法选出被标注次数最多的主题作为最终标签。
  - 如果每个主题被标注的次数相同，则统计这类样本的数量，如果数量较少例如小于总数据集的1%则可以作为outlier删除该样本；如果样本数量较多则随机选择一个主题作为该样本的最终标签
  - 需要对全体数据集都进行统一标签的操作
  - 代码实现见 `sandplay_homework/SandPlayPreProcess/unify_labels.py`
- 设计模型：本次实验需要设计6个模型，包括基于预训练Vit-B/16的分类模型、基于预训练resnet-50的分类模型、不使用预训练参数的resnet-50分类模型、自主设计的卷积神经网络模型、KNN模型、K-means模型
  - 模型设计使用torch，避免从头造轮子。
  - 代码实现见 `sandplay_homework/model_utils/model_base.py`
- 设计数据结构：
  - 本次实验需要自主设计基于torch的 `Dataset`和 `DataLoader`
  - 代码实现见 `sandplay_homework/data/datasets/sandplay_dataset.py`，`sandplay_homework/data/build.py`
- 数据预处理设计：预训练Vit-B/16和预训练resnet分别需要设计对应的transforms
  - 代码实现见 `sandplay_homework/data/transforms/build_transforms.py`
- 选择优化器：
  - 代码实现见 `sandplay_homework/solver/build_optimizer.py`
- 实现训练和推理逻辑：
  - 代码实现见 `sandplay_homework/engine/trainer.py`，`sandplay_homework/engine/inference.py`
- 实现训练与测试的主程序：
  - 代码实现见 `sandplay_homework/tools/train_net.py`，`sandplay_homework/tools/test_net.py`
- 实现log和学习过程的可视化监控
  - 代码实现见 `sandplay_homework/utils/logger.py`

### 结果与分析

<!-- 简单比较实验组和对照的结果，理解所控制的变量和实验观测指标的关系，检验前面的实验预期，得出有效结论。-->

- 在进行统一标签的步骤时发现309个数据中有13个图像没有专家标注，因此这13张图可以看作无效样本；62个图像每个专家标注都互不相同，因此这62个图像不具备有参考价值的标签，可以看作噪声样本；19个图像数量最多的专家标注是6个主题之外的词，因此这19张图可以看作偏差样本。这些样本对于训练会产生不利影响，因此本实验考虑将其剔除。清理过后剩下215个样本，其中训练样本181个，测试样本34个
- 本次实验将训练数据按照9：1进一步划分为训练集和验证集，即训练集163张图，验证集18张图。后续可以尝试使用K-fold validation
