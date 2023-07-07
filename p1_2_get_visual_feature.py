# -*- coding:utf-8 -*-
# author: Xiaokun Feng
# e-mail: fengxiaokun2022@ia.ac.cn
# datetime:2023/3/23 14:57
"""
description: 举例说明 使用 model_utils 来使用 沙盘预训练视觉模型 的方法
            注：沙具多标签多分类 并不是我们所关心的任务，只是借助此任务来得到一个 关于沙盘图像的预训练模型
            我们希望 使用 此预训练模型 来服务于 最终的 沙盘主题识别
"""
import json

from model_utils.model_base import Theme_Classification_model_v1
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from PIL import Image
BICUBIC = InterpolationMode.BICUBIC

img_transform = Compose([
            Resize((224,224), interpolation=BICUBIC),
            ToTensor(),
            Normalize((0.5077,0.7075,0.6756), (0.2336,0.2158,0.3488))

        ])

if __name__ == "__main__":
    # 1.导入模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    my_model = Theme_Classification_model_v1(device)
    my_model.to(device)

    my_model.load_state_dict(torch.load("/raid/ckh/sandplay_homework/model_utils/sand_object_classification.pkl"))
    my_model.eval()

    # for name, param in my_model.named_parameters():
    #     print(f"Layer: {name}")

    # 2.模型 推理
    img_path = "/raid/ckh/sandplay_homework/resource/homework_sand_label_datasets/20201109101336_142/BireView.png"
    img_data = Image.open(img_path)
    img_data = img_transform(img_data)
    img_data = img_data.to(device, dtype=torch.float)
    img_data = img_data.unsqueeze(dim=0)
    y_out = my_model(img_data)

    # 3.根据 resource/theme_label_infor.json 来确定 最终 分类所得的沙具名称
    json_path = "/raid/ckh/sandplay_homework/resource/sanders_onehot_label.json"
    with open(json_path, 'r', encoding='utf-8') as fp:
        sander_name_dict = json.load(fp)
        fp.close()

    id2name_dict = dict(zip(sander_name_dict.values(), sander_name_dict.keys()))

    sander_namelist = []
    y_out = y_out[0,:]
    print(y_out.shape)
    for y_index,y_data_item in enumerate(y_out):
        if y_data_item > 0.5:
            sander_namelist.append(id2name_dict[y_index])

    print("judge result:",sander_namelist)

