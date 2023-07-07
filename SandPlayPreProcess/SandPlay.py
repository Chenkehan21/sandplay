# -*- coding:utf-8 -*-
# author: Xiaokun Feng
# e-mail: fengxiaokun2022@ia.ac.cn
# datetime:2023/3/23 12:06
"""
description: 对 沙盘文件 进行预处理

"""
import numpy as np
import math
import cv2
from sklearn.cluster import  DBSCAN
import json
import sys
sys.path.append("./")

# 定义一些宏变量
RIVER_THROHOLD = 2  # 对于高度图，对于像素小于 此值的点，认为是 河流区域
DUNE_THROHOLD = 20  # 对于高度图，对于像素大于 此值的点，认为是 沙丘区域
HEIGHTMAP_SIZE = 50 # 河流/沙丘高度图 尺寸 为50


# 沙具类
# 定义 每一个沙具 的基本属性信息
class SanderData():
    def __init__(self):
        self.sander_name = ""
        self.sander_name_c = ""

        self.sander_class = 0
        self.xy_data = [0]*2
        self.wh_data = [0]*2
        self.angle_data = 0
        self.scale_data = 0
        self.dump_flag = 0  # 倾倒
        self.handstand_flag = 0  # 倒立
        # 新增 属性信息
        self.attr_1 = 0
        self.attr_2 = 0
        self.attr_3 = 0
        self.attr_4 = 0
        self.attr_5 = 0    #表示沙具是否为原型物
        self.attr_6 = 0
        self.attr_7 = 0
        self.item_type = 0
        self.sub_item_type = ""
        # self.scene_type = ""
        self.attr_link_for_division_judge = 0
        self.attr_link_for_link_judge = 0
        self.attr_link_for_link_judge0 = 0
        self.centripetal_judge = 0
        self.attr_for_ignore = 0
        self.attr_for_threat = 0
        self.attr_for_obstruct = 0
        self.attr_for_dis_res = 0
        self.attr_for_dump = 0
        self.attr_for_empty = 0
        self.attr_for_energy = 0
        # 待计算 变量
        self.rad = 0
        self.rad_s = 0
        self.wh_s = 0


    def construct(self,data):
        # data = (name,class,x_data,y_data,angle,dump,handstand,scale,w_data,h_data)
        self.sander_name = data[0]
        self.sander_class = data[1]
        self.xy_data[0] = data[2]
        self.xy_data[1] = data[3]
        self.angle_data = data[4]
        self.dump_flag = data[5]
        self.handstand_flag = data[6]
        self.scale_data = data[7]
        self.wh_data[0] = data[8]
        self.wh_data[1] = data[9]
        self.rad = ((self.wh_data[0]**2 + self.wh_data[1]**2)**0.5)/2
        self.rad_s = math.pi*(self.rad**2)
        self.wh_s = self.wh_data[0]*self.wh_data[1]

# 地形类
# 定义 河流/沙丘 的基本属性信息/构造方法
class RiverDuneData():
    def __init__(self):
        # 地形信息
        self.include_river_flag = False
        self.include_dune_flag = False
        self.river_data = []  #list中每一个元素，表示的是 每一簇 river 所对应的xy_data 集合
        self.river_map = []
        self.river_bbox_data = []  # x,y,w,h
        self.dune_data = []
        self.dune_map = []
        self.dune_bbox_data = []  # x,y,w,h
    def get_river_dune_bbox_infor(self,height_map_path):
        # 默认river_map的尺寸为50*50
        river_save_path = height_map_path[:-4] + '_river_map.png'
        dune_save_path = height_map_path[:-4] + '_dune_map.png'
        #-----------------pos_1.对数据进行预处理----------------------
        heightmap_data = cv2.imread(height_map_path)
        heightmap_data = heightmap_data[:, :, 0]
        # 筛选
        river_map = np.zeros(heightmap_data.shape)
        dune_map = np.zeros(heightmap_data.shape)
        for i in range(heightmap_data.shape[0]):
            for j in range(heightmap_data.shape[1]):
                if heightmap_data[i, j] < RIVER_THROHOLD:
                    river_map[i, j] = 255
                if heightmap_data[i,j] > DUNE_THROHOLD:
                    dune_map[i,j] = 255

        # 缩放处理 50*50
        river_map = cv2.resize(river_map, (50, 50), interpolation=cv2.INTER_AREA)
        dune_map = cv2.resize(dune_map, (50, 50), interpolation=cv2.INTER_AREA)
        # 进行二值化处理：
        for i in range(river_map.shape[0]):
            for j in range(river_map.shape[1]):
                if river_map[i, j] > 100:
                    river_map[i, j] = 255
                else:
                    river_map[i, j] = 0
        for i in range(dune_map.shape[0]):
            for j in range(dune_map.shape[1]):
                if dune_map[i, j] > 100:
                    dune_map[i, j] = 255
                else:
                    dune_map[i, j] = 0

        self.river_map = river_map
        self.dune_map = dune_map
        #-----------------pos_2.获取river相关数据----------------------
        cv2.imwrite(river_save_path, river_map)
        # pos_1)先根据river_map_data求出 x y的范围
        river_map_points_num = int(np.sum(river_map)/255)
        # print('river_map_points_num: ',river_map_points_num)
        if river_map_points_num < 80:
            self.include_river_flag = False
        else:
            self.include_river_flag = True
            river_map_data = np.zeros([river_map_points_num,2])
            k = 0
            for i in range(river_map.shape[0]):
                for j in range(river_map.shape[1]):
                    if river_map[i, j] == 255:
                        river_map_data[k,0] = i
                        river_map_data[k,1] = j
                        k = k + 1

            model = DBSCAN(eps=2, min_samples=3)
            # 模型拟合与聚类预测
            yhat = model.fit_predict(river_map_data)
            # 检索唯一群集
            clusters, index, counts = np.unique(yhat, return_inverse=True, return_counts=True)
            # 统计每一个河流簇，对每一个河流的信息进行记录
            river_bbox_data = []
            river_data = []
            for cluster_num in clusters:
                if cluster_num >-1:
                    row_index = np.where(yhat == cluster_num)
                    temp_river_map_data = river_map_data[row_index[0],:]
                    river_data.append(temp_river_map_data)
                    river_bbox_data.append(self.get_bbox(temp_river_map_data))
            self.river_bbox_data = river_bbox_data
            self.river_data = river_data
        # -----------------pos_3.获取dune 相关数据----------------------
        cv2.imwrite(dune_save_path, dune_map)
        # pos_1)先根据dune_map_data求出 x y的范围
        dune_map_points_num = int(np.sum(dune_map) / 255)
        # print('dune_map_points_num: ',dune_map_points_num)
        if dune_map_points_num < 50:
            self.include_dune_flag = False
        else:
            self.include_dune_flag = True
            dune_map_data = np.zeros([dune_map_points_num, 2])
            k = 0
            for i in range(dune_map.shape[0]):
                for j in range(dune_map.shape[1]):
                    if dune_map[i, j] == 255:
                        dune_map_data[k, 0] = i
                        dune_map_data[k, 1] = j
                        k = k + 1

            model = DBSCAN(eps=2, min_samples=3)
            # 模型拟合与聚类预测
            yhat = model.fit_predict(dune_map_data)
            # 检索唯一群集
            clusters, index, counts = np.unique(yhat, return_inverse=True, return_counts=True)
            # 统计每一个dune簇，对每一个河流的信息进行记录
            dune_bbox_data = []
            dune_data = []
            for cluster_num in clusters:
                if cluster_num > -1:
                    row_index = np.where(yhat == cluster_num)
                    temp_dune_map_data = dune_map_data[row_index[0], :]
                    dune_data.append(temp_dune_map_data)
                    dune_bbox_data.append(self.get_bbox(temp_dune_map_data))
            self.dune_bbox_data = dune_bbox_data
            self.dune_data = dune_data

    def get_bbox(self,map_data):
        data = []
        x_max0, y_max0 = np.max(map_data, axis=0)
        x_min0, y_min0 = np.min(map_data, axis=0)
        # pos_2) 对其进行缩放和转化处理
        x_max1 = -64 * y_min0 / 50 + 32
        x_min1 = -64 * y_max0 / 50 + 32
        y_min1 = 52 * x_min0 / 50 - 26
        y_max1 = 52 * x_max0 / 50 - 26

        data.append((x_min1 + x_max1) / 2)
        data.append((y_min1 + y_max1) / 2)
        data.append((x_max1 - x_min1))
        data.append((y_max1 - y_min1))
        river_data_2 = [x_min1, y_min1, x_max1, y_max1]
        return river_data_2

# 沙盘类
class SandPlayData():
    def __init__(self,data_root):
        """"
        Parameters
        ----------
        data_root : 沙盘文件的路径信息
        """
        # 1.先初始化相关参数
        self.data_root = data_root
        file_path_list = self.data_root.split('/')
        self.sander_name = file_path_list[-1]

        self.sanders_infor = []
        self.river_dune_infor = RiverDuneData()
        self.theme_dict = {}
        self.theme_dict_init()

        # 2.完成数据的 导入
        # 得到 self.sanders_infor 和 self.river_dune_infor
        self.data_load()

        self.sander_num = len(self.sanders_infor)

        self.xy_data = np.zeros([self.sander_num,2])
        self.size_data = np.zeros([self.sander_num,2])
        self.rad_data  = np.zeros([self.sander_num])

        k = 0
        for sand_infor in self.sanders_infor:
            self.xy_data[k,0] = sand_infor.xy_data[0]
            self.xy_data[k, 1] = sand_infor.xy_data[1]
            self.size_data[k,0] = sand_infor.wh_data[0]
            self.size_data[k,1] = sand_infor.wh_data[1]
            self.rad_data[k] = sand_infor.rad
            k = k + 1

        print("sander len: ",len(self.sanders_infor))
        # 3. 进行主题判断
        # TODO: 编写完成各个主题的 判断程序
        self.theme_judge()
        # print(self.theme_dict)

    def theme_dict_init(self):
        """
        initialize the theme infor about the sandplay
        """
        self.theme_dict["integration"] = 0.0        # 整合
        self.theme_dict["flow"] = 0.0               # 流动
        self.theme_dict["link"] = 0.0               # 联结

        self.theme_dict["split"] = 0.0              # 分裂
        self.theme_dict["chaos"] = 0.0              # 混乱
        self.theme_dict["empty"] = 0.0              # 空洞

    def data_load(self):
        """
        根据 stinfor__.txt文件 以及 高度图 得到沙盘相关数据信息
        """
        # 1. 先获取相关路径信息
        file_name_path = self.data_root + '/' + 'STinfo_' + self.sander_name + '.txt'
        height_map_path = self.data_root + '/heightmap.png'
        data = []
        with open(file_name_path, "r") as file:
            data = file.readlines()
        # 获取 json 文件对应的列表文件

        json_path = "SandPlayPreProcess/SandersInfor_dict.json"
        with open(json_path, 'r', encoding='utf8') as fp:
            json_dict = json.load(fp)
            fp.close()

        sander_data = []
        for line in data:
            temp_sander_data = SanderData()
            line = line[:-1]
            line = line.split(";")
            temp_sander_data.construct(data_generate(line))
            sander = json_dict[temp_sander_data.sander_name]

            temp_sander_data.attr_1 = sander['attr_1']
            temp_sander_data.attr_2 = sander['attr_2']
            temp_sander_data.attr_3 = sander['attr_3']
            temp_sander_data.attr_4 = sander['attr_4']
            temp_sander_data.attr_5 = sander['attr_5']
            temp_sander_data.attr_6 = sander['attr_6']
            temp_sander_data.attr_7 = sander['attr_7']
            temp_sander_data.item_type = sander['item_type']
            temp_sander_data.sub_item_type = sander['sub_item_type']
            # temp_sander_data.scene_type = sander['scene_type']
            temp_sander_data.attr_link_for_division_judge = sander['attr_link_for_division_judge']
            temp_sander_data.attr_link_for_link_judge = sander['attr_link_for_link_judge']
            temp_sander_data.attr_link_for_link_judge0 = sander['attr_link_for_link_judge0']
            temp_sander_data.centripetal_judge = sander['centripetal_judge']
            temp_sander_data.attr_for_ignore = sander['attr_for_ignore']
            temp_sander_data.attr_for_threat = sander['attr_for_threat']
            temp_sander_data.attr_for_obstruct = sander['attr_for_obstruct']
            temp_sander_data.attr_for_dis_res = sander['attr_for_dis_res']
            temp_sander_data.attr_for_dump = sander['attr_for_dump']
            temp_sander_data.attr_for_empty = sander['attr_for_empty']
            temp_sander_data.attr_for_energy = sander['attr_for_energy']
            sander_data.append(temp_sander_data)

        self.river_dune_infor.get_river_dune_bbox_infor(height_map_path)
        self.sanders_infor = sander_data

    def theme_judge(self):
        self.integration_judge()
        self.flow_judge()
        self.link_judge()

        self.split_judge()
        self.chaos_judge()
        self.empty_judge()



    # ---------------------------对各个主题判断方法的封装-----------------
    def integration_judge(self):
        """
        整合主题的判断
        :return:
        """
        # TODO:整合主题的判断
        pass


    def chaos_judge(self):
        """
        混乱主题的判断
        :return:
        """
        # TODO:混乱主题的判断
        pass

    def flow_judge(self):
        """
        流动主题的判断
        :return:
        """
        # TODO:流动主题的判断
        pass

    def link_judge(self):
        """
        联结主题的判断
        :return:
        """
        # TODO:联结主题的判断
        pass

    def split_judge(self):
        """
        分裂主题的判断
        :return:
        """
        # TODO:分裂主题的判断
        pass

    def empty_judge(self):
        """
        空洞主题的判断
        :return:
        """
        # TODO:空洞主题的判断
        pass

def data_generate(line):
    name = line[0]
    calss = int(line[1])
    xy_data = line[2].split(",")
    x_data = float(xy_data[0])
    y_data = float(xy_data[1])
    angle = float(line[3])
    dump = float(line[4])
    handstand = float(line[5])
    scale = float(line[6])
    wh_data = line[7].split(",")
    w_data = float(wh_data[0])
    h_data = float(wh_data[1])
    line_tuple = (name,calss,x_data,y_data,angle,dump,handstand,scale,w_data,h_data)
    return line_tuple
