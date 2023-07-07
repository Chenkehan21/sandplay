# -*- coding:utf-8 -*-
# author: Xiaokun Feng
# e-mail: fengxiaokun2022@ia.ac.cn
# datetime:2023/3/23 12:26
"""
description: 举例说明 使用 SandPlayPreProcess 来读取沙盘文件 基本信息的方法
"""
from SandPlayPreProcess.SandPlay import SandPlayData

if __name__ == "__main__":
    # 1.先指定 一个沙盘文件 的路径
    sandplay_file_path = "/raid/ckh/sandplay_homework/resource/homework_sand_label_datasets/20201109101336_142"

    # 2.实例化一个 与此沙盘文件 对应的 沙盘实例
    # 此实例 包含了 此沙盘文件 含有的 基本信息

    sandplay_infor = SandPlayData(data_root=sandplay_file_path)

    # 3.使用举例：输出此沙盘中的各个沙具的 名字信息，和坐标信息
    sanders_infor = sandplay_infor.sanders_infor
    for sander_item in sanders_infor:
        print(sander_item.sander_name,": ",sander_item.xy_data)
