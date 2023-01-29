"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob
import torch.utils.data

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train  # 是否为训练集
        self.data_aug_conf = data_aug_conf  # 数据增强配置
        self.grid_conf = grid_conf  # 网格配置

        self.scenes = self.get_scenes()  # 得到scene名字的列表list: [scene-0061, scene-0103,...]
        self.ixes = self.prepro()  # 得到属于self.scenes的所有sample

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        # dx=[0.5,0.5,20]
        # bx=[-49.75,-49.75,0]
        # nx=[200,200,1]
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()  # 转化成numpy

        self.fix_nuscenes_formatting()  # If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the
        # file paths stored in the nuScenes object

        print(self)

    # fix_nuscenes_formatting() 调整ncscenes数据格式 (被类初始化函数调用)
    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (
                        rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    # get_scenes() 根据 self.nusc.version 场景分为训练集和验证集(被类初始化函数调用）
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]  # 根据 self.nusc.version 场景分为训练集和验证集，得到的是场景名字的list: [scene-0061,
        # scene-0103,...]

        return scenes

    def prepro(self):  # 将self.scenes中的所有sample取出并依照 scene_token和timestamp排序  (被类初始化函数调用)
        """
        存储当前划分的所有样本，且按场景和时间戳排序
        获取一系列样本, 包含了传感器采集到的信息、标注信息等等。形如：
        { 'token': 'ca9a282c9e77460f8360f564131a8af5',
          'timestamp': 1532402927647951,
          'prev': '',
          'next': '39586f9d59004284a7114a68825e8eec',
          'scene_token': 'cc8c0bf57f984915a77078b10eb33198',
          'data': {'RADAR_FRONT': '37091c75b9704e0daa829ba56dfa0906',
                   'RADAR_FRONT_LEFT': '11946c1461d14016a322916157da3c7d',
                   'RADAR_FRONT_RIGHT': '491209956ee3435a9ec173dad3aaf58b',
                   'RADAR_BACK_LEFT': '312aa38d0e3e4f01b3124c523e6f9776',
                   'RADAR_BACK_RIGHT': '07b30d5eb6104e79be58eadf94382bc1',
                   'LIDAR_TOP': '9d9bf11fb0e144c8b446d54a8a00184f',
                   'CAM_FRONT': 'e3d495d4ac534d54b321f50006683844',
                   'CAM_FRONT_RIGHT': 'aac7867ebf4f446395d29fbd60b63b3b',
                   'CAM_BACK_RIGHT': '79dbb4460a6b40f49f9c150cb118247e',
                   'CAM_BACK': '03bea5763f0f4722933508d5999c5fd8',
                   'CAM_BACK_LEFT': '43893a033f9c46d4a51b5e08a67a1eb7',
                   'CAM_FRONT_LEFT': 'fe5422747a7d4268a4b07fc396707b23'},
          'anns': ['ef63a697930c4b20a6b9791f423351da',
                   '6b89da9bf1f84fd6a5fbe1c3b236f809',
                   ...]},
        """
        samples = [samp for samp in self.nusc.sample]  # self.nusc.sample 可以拿到所有的关键帧数据

        # 删除不在划分（训练集或者测试集）中的样本
        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        # 按场景、时间戳排序（只是为了使按时间顺序更容易）
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    # sample_augmentation() 获取对图片进行数据增强的参数(被get_image_data()函数调用)
    def sample_augmentation(self):
        """
        返回数据增强的相关设置
        resize：         随机缩放比率
        resize_dims：    缩放后的尺度
        crop：           随机裁剪：(crop_w, crop_h, crop_w + fW, crop_h + fH) 分别为左上和右下的坐标
        flip：           是否左右反转 bool
        rotate：         随机旋转的角度
        """
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']  # (900,1600) # 图像的原始尺寸
        fH, fW = self.data_aug_conf['final_dim']  # (128, 352)，表示变换之后最终的图像大小, 最终需要的尺寸
        if self.is_train:  # 训练集数据增强
            # 函数原型： numpy.random.uniform(low,high,size)
            # 功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
            # ----------------- 随机选择比率缩小 随机缩放图片大小----------------
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims

            # ----------------- 随机裁剪图片 ----------------
            # crop_h：高度需要裁剪的大小；crop_w：宽度需要裁剪的大小，如果newW <= fw则为0；
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            # ----------------- 随机翻转图片 ----------------
            flip = False   # 0.5的概率会采用翻转
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True

            # ----------------- 随机旋转图片 ----------------
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])   # 旋转的角度
        else:  # 测试集数据增强
            resize = max(fH / H, fW / W)  # 缩小的倍数取二者较大值: 0.22
            resize_dims = (int(W * resize), int(H * resize))  # 保证H和W以相同的倍数缩放，resize_dims=(352, 198)
            newW, newH = resize_dims  # (352,198)
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH  # 48
            crop_w = int(max(0, newW - fW) / 2)  # 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)  # (0, 48, 352, 176)，对应裁剪的左上角和右下角的坐标
            # 验证时关闭随机翻转与旋转
            flip = False  # 不翻转
            rotate = 0  # 不旋转
        return resize, resize_dims, crop, flip, rotate

    # get_image_data 得到图像数据以及各种参数信息(被 SegmentationData 类中的__getitem__函数调用)
    def get_image_data(self, rec, cams):
        imgs = []               # 经过数据增强、归一化与标准化后的图像tensor
        rots = []               # 相机坐标系相对的旋转参数  # 相机坐标系到自车坐标系的旋转矩阵
        trans = []              # 相机坐标系相对的平移参数  # 相机坐标系到自车坐标系的平移向量
        intrins = []            # 相机的内参
        post_rots = []          # 图像增强旋转对应的单应旋转矩阵  # 数据增强的像素坐标旋转映射关系
        post_trans = []         # 图像增强平移对应的单应平移矩阵  # 数据增强的像素坐标平移映射关系
        # 从每一个选择的相机数据中读取数据
        for cam in cams:
            """ 
            rec 表示当前选取的样本；参考 prepro 的数据格式
            读取图片信息, samp 图片信息如下：
            {'token': 'd6206af179d44816b034e3cbd8ab5eea',
             'sample_token': 'cd9964f8c3d34383b16e9c2997de1ed0',
             'ego_pose_token': 'd6206af179d44816b034e3cbd8ab5eea',
             'calibrated_sensor_token': '19eeff32f00f414fb9489ccab9c6c2c7',
             'timestamp': 1535657108254799,
             'fileformat': 'jpg',
             'is_key_frame': True,
             'height': 900,
             'width': 1600,
             'filename': 'samples/CAM_FRONT_LEFT/n008-2018-08-30-15-16-55-0400__CAM_FRONT_LEFT__1535657108254799.jpg',
             'prev': '',
             'next': '44398719bd8f4a94b36f89d8da649adf',
             'sensor_modality': 'camera',
             'channel': 'CAM_FRONT_LEFT'}
            """
            samp = self.nusc.get('sample_data', rec['data'][cam])           # 根据相机通道选择对应的sample_data
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])    # 图片的路径
            img = Image.open(imgname)   # 读取图像 1600 x 900
            post_rot = torch.eye(2)     # 增强前后像素点坐标的旋转对应关系
            post_tran = torch.zeros(2)  # 增强前后像素点坐标的平移关系

            """ 读取该图片的相关信息，包括旋转，平移及相机内参, 如：
            {'token': '19eeff32f00f414fb9489ccab9c6c2c7',
             'sensor_token': 'ec4b5d41840a509984f7ec36419d4c09',
             'translation': [1.5752559464, 0.500519383135, 1.50696032589],
             'rotation': [0.6812088525125634,
                          -0.6687507165046241,
                          0.2101702448905517,
                          -0.21108161122114324],
             'camera_intrinsic': [[1257.8625342125129, 0.0, 827.2410631095686],
                                  [0.0, 1257.8625342125129, 450.915498205774],
                                  [0.0, 0.0, 1.0]]}
            """

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])  # 相机record
            intrin = torch.Tensor(sens['camera_intrinsic'])  # 相机内参
            # 相机坐标系相对于ego坐标系的旋转矩阵
            # nus中的rotation 是四元数，这个把四元数转换为旋转矩阵
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])  # 相机坐标系相对于ego坐标系的平移矩阵

            # ------------------------ 数据增强 -------------------------
            # 增强设置（调整大小、裁剪、水平翻转、旋转）
            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()  # 获取数据增强的参数
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate,
                                                       )  # 进行数据增强: resize->crop,并得到增强前后像素点坐标的对应关系

            # for convenience, make augmentation matrices 3x3
            # 为方便起见，将增强矩阵设为 3x3
            # 写成3维矩阵的形式
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))  # 标准化: ToTensor, Normalize 3,128,352
            intrins.append(intrin)  # 3,3
            rots.append(rot)  # 3,3
            trans.append(tran)  # 3,
            post_rots.append(post_rot)  # 3,3
            post_trans.append(post_tran)  # 3,

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))  # 使用torch.stack组装到一起

    # get_lidar_data 获取雷达数据
    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                             nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    '''
        世界坐标系下的标注信息转化到车辆坐标系下，得到对应分割的BEV视图。
        得到自车坐标系相对于地图全局坐标系的位姿 (被SegmentationData 中的__getitem__调用)
    '''
    def get_binimg(self, rec):
        # 得到自车坐标系相对于地图全局坐标系的位姿
        """ 读取车辆姿态信息, 如：
        {'token': '2b2d98f0daec4fdeb6716d1f6b546a5c',
         'timestamp': 1535489298047428,
         'rotation': [0.3559189826878399,
                      0.0018057365272423622,
                      -0.009148368803714158,
                      0.934470290820569],
         'translation': [1316.378185650685, 1038.5936853755643, 0.0]}
        """
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])  # 自车的位置
        trans = -np.array(egopose['translation'])       # 平移
        rot = Quaternion(egopose['rotation']).inverse   # 旋转
        img = np.zeros((self.nx[0], self.nx[1]))        # 初始化图像, 200 x 200的网格
        for tok in rec['anns']:                         # 遍历该sample的每个annotation token
            # 读取当前标注信息
            """如：
            {'token': 'f5f9b9db580d412083fd18ee4fda0b8e',
             'sample_token': '2021b6b367984ad7b18464a550d0ae8d',
             'instance_token': '19804c352c0a4767b61b8d0709d1db99',
             'visibility_token': '4',
             'attribute_tokens': ['ab83627ff28b465b85c427162dec722f'],
             'translation': [1294.421, 1031.646, 0.507],
             'size': [0.552, 0.918, 1.739],
             'rotation': [0.9380281279379362, 0.0, 0.0, 0.3465591308813702],
             'prev': '70885047ed1c4e90924bf304c830501b',
             'next': 'd3ba83018b4c45b0b3523bab752b08c3',
             'num_lidar_pts': 0,
             'num_radar_pts': 0,
             'category_name': 'human.pedestrian.adult'}
            """
            inst = self.nusc.get('sample_annotation', tok)  # 找到该annotation
            # add category for lyft
            # 为 lyft 添加类别， 如果标注为 vehicle 则跳过此条标注信息
            if not inst['category_name'].split('.')[0] == 'vehicle':  # 只关注车辆
                continue

            """
            Box() 为表示 3d 框的简单数据类，包括标签、分数和速度。
            使用说明
                :param center:          以 x, y, z 形式给出的框的中心。
                :param size:            框的宽度、长度、高度。
                :param orientation：    box方向。
                :param label：          整数标签，可选。
                :param score：          分类分数，可选。
                :param velocity:        x, y, z 方向的box速度。
                :param name:            box名称，可选。 可以使用例如 for 表示类别名称。
                :param token：          来自 DB 的唯一字符串标识符。
            """

            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))  # 参数分别为center, size, orientation
            box.translate(trans)  # 平移, 将box的center坐标从全局坐标系转换到自车坐标系下
            box.rotate(rot)       # 旋转, 将box的center坐标从全局坐标系转换到自车坐标系下

            # box.bottom_corners() 返回box的四个底角。前两个面朝前，后两个面朝后。shape: [3, 4]
            pts = box.bottom_corners()[:2].T  # 三维边界框取底面的四个角的(x,y)值后转置, 4x2
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)  # 将box的实际坐标对应到网格坐标，同时将坐标范围[-50,50]平移到[0,100]
            pts[:, [1, 0]] = pts[:, [0, 1]]  # 把(x,y)的形式换成(y,x)的形式
            cv2.fillPoly(img, [pts], 1.0)  # 在网格中画出box

        return torch.Tensor(img).unsqueeze(0)  # 转化为Tensor 1x200x200

    # 随机选择摄像机通道， 也算数据增强的一种
    def choose_cams(self):
        '''
            numpy.random.choice(a, size=None, replace=True, p=None)
            从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
            replace:True表示可以取相同数字，False表示不可以取相同数字
            数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
        '''

        # ---------------------- 这里表示随机选取 'Ncams' 个相机的图片 ---------------------
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            # 选择全部的相机通道
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)  # 获取雷达数据， nsweeps：先前帧数
        binimg = self.get_binimg(rec)

        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]  # 按索引取出sample

        cams = self.choose_cams()  # 对于训练集且data_aug_conf中Ncams<6的，随机选择摄像机通道，否则选择全部相机通道
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)  # 读取图像数据、相机参数和数据增强的像素坐标映射关系
        # imgs: 6,3,128,352  图像数据
        # rots: 6,3,3  相机坐标系到自车坐标系的旋转矩阵
        # trans: 6,3  相机坐标系到自车坐标系的平移向量
        # intrins: 6,3,3  相机内参
        # post_rots: 6,3,3  数据增强的像素坐标旋转映射关系
        # post_trans: 6,3  数据增强的像素坐标平移映射关系
        binimg = self.get_binimg(rec)  # 得到rec中所有box相对于车辆的box坐标的平面投影图, 1x200x200
        # binimg中在box内的位置值为1，其他位置的值为0

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg

# worker_rnd_init 获取随机种子（被compile_data 中的Dataloader函数调用）
def worker_rnd_init(x):  # x是线程id
    np.random.seed(13 + x)

'''
    compile_data 主要做的工作
    1、首先是调用nuscenes.nuscenes.NuScenes 库构建了一个nusc的数据集
    2、然后把nusc作为参数传入parser() 中构建数据解析器traindata和valdata 。
    3、其中parser 根据输入的参数parser_name有两种选择，一个是VizData,一个是SegmentationData (这两个都是继承自定义的NuscData的Dataset类)
    然后traindata和valdata 再把这两个参数传入torch.utils.data.DataLoader 构建了训练集和测试集的数据加载器，并返回。
'''

def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)  # 加载nuScenes数据集
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]  # 根据传入的参数选择数据解析器

    # 加载训练集 与 验证集
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                     grid_conf=grid_conf)

    # 训练数据加载器
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)  # 给每个线程设置随机种子

    # 验证数据加载器
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
