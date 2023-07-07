# 概述

fengxiaokun2022@ia.ac.cn

此项目对应作业一（图像主题识别）。包含了所需数据，以及一些基本的预处理程序。

文件整体目录为：

- resource: 作业所需用到的相关资源，如数据集源文件，文件名称信息等
- SandPlayPreProcess：沙盘文件预处理程序
- model_utils：基于沙盘图像的视觉预训练模型

对各个文件作进一步说明：

## resource

-     homework_sand_label_datasets: 本次作业所使用到的数据集，此目录下的每一个文件夹对应一个数据样本。每个
  数据样本包含如下文件：

  1）BireView.png 表示沙盘图像

  2）heightmap.png 表示沙盘中 的 地势/河流信息

  3）STinfo_xx.txt 表示沙盘中出现的所有沙具信息；其中，每一行表示一个沙具，包含的沙具信息有
  （沙具名称，沙具类别，,x坐标，y坐标，xy平面的旋转角度，是否倾斜，是否颠倒，缩放比例，沙具长度，沙具宽度）

  4）theme_label_infor.json表示主题标注信息；以列表的形式给出，列表中的每一个元素表示（不同）标注者判断的主题结果，以及相应的原因说明。
  我们关注的6个主题，及其对应的正样本数量为：{'整合': 162, '流动': 221, '联结': 192,'分裂': 130, '混乱': 113, '空洞': 133}

  tips:可以结合不同主题判定结果所对应的原因，来设计相应的主题判断程序

  注：对于一个沙盘样本，BireView.png以视觉图像的形式，包含了其所有的信息；heightmap.png和STinfo_xx.txt文件一起，以另一种形式也包含了沙盘的所有信息
- file_namelist_infor.json：以字典的形式存储沙盘样本名称信息。其中,"all_file"表示所有的样本文件名称（共409个）；
  "train_file"与"test_file"表示划分的训练集样本（260个）和测试集样本（49个）。

  注：应基于train_file中的文件来进行模型设计；使用test_file中的文件来测试模型的性能。提交作业时，需给出模型在训练集和测试集上的性能测量指标。
- sanders_onehot_label.json：整个沙盘环境中，所涉及的所有沙具名称。每个沙具对应一个唯一的编号，在下文 沙具多标签多分类模型的设计中会有所涉及。

## SandPlayPreProcess

沙盘预处理程序库，将沙盘文件 读取和保存成特定的数据格式。可在此基础上进行主题识别程序的设计。

使用说明请参照 p1_1_load_sandplay_infor.py

## model_utils

由于数据集的规模较小（仅有409组有标注沙盘样本），不便于直接使用深度学习的方法。
这里提供了一个针对BireView.png图像的沙盘视觉预训练模型，可使用此模型得到的 特征图（网络后几层），来进行
后续主题识别模型的设计。

对于此模型，其所实现的任务是关于沙盘中所出现沙具的多标签多分类任务，其输入数据为BireView.png图像，输出数据为所有沙具（494个）的分类概率（概率大于0.5，视为沙盘中存在此沙具）。
网络模型由 clip视觉编码模块（具体为 "ViT-B/16"） + 全连接分类层 组成，详情见CLIP（https://github.com/openai/CLIP）
以及model_base.py文件

此模型是在另外一个中等规模（5k） 未标注主题的 数据集（未提供）上训练得到，沙具的多标签多分类准确率可以达到80%（由于 沙具长尾分布的原因，
对于那些出现频率很少的沙具识别效果不佳，但对经常出现的绝大多数沙具，都能进行识别）。模型的权重为 sand_object_classification.pkl。

模型的使用请参照 p1_2_get_visual_feature.py

注：沙具多标签多分类 并不是我们所关心的任务，只是借助此任务来得到一个 关于沙盘图像的预训练模型，
我们希望 使用 此预训练模型 来服务于 最终的 沙盘主题识别
