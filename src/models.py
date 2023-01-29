"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


'''
    Up类:  上采样
    上采样（被CamEncode类和BEVEncode类中的初始化函数调用）
'''
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)  # 上采样 BxCxHxW->BxCx2Hx2W

        self.conv = nn.Sequential(  # 两个3x3卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True使用原地操作，节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 对x1进行上采样
        x1 = torch.cat([x2, x1], dim=1)  # 将x1和x2 concat 在一起
        return self.conv(x1)

# 提取图像特征进行图像编码 （被LiftSplatShoot类中的初始化函数调用）
class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D  # 深度上的网格数：41
        self.C = C  # 图像特征维度：64

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征

        self.up1 = Up(320+112, 512)  # 上采样模块，输入输出通道分别为320+112和512
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)  # 1x1卷积，变换维度

    def get_depth_dist(self, x, eps=1e-20):  # 对深度维进行softmax，得到每个像素不同深度的概率
        return x.softmax(dim=1)

    # 提取带有深度的特征
    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)  # 使用efficientnet提取特征  x: 24 x 512 x 8 x 22
        # Depth
        # 1x1卷积变换维度， x: 24x105x8x22 = 24x(C+D)xfHxfW
        x = self.depthnet(x)

        '''
        第二个维度的前D个作为深度维(把连续的深度值离散化)，进行softmax  depth: 24 x 41 x 8 x 22
        第二个维度后64个的数据作为图像特征
        '''
        depth = self.get_depth_dist(x[:, :self.D])

        '''
        将特征通道维和通道维利用广播机制相乘 
        depth.unsqueeze(1) -> torch.Size([24, 1, 41, 8, 22])
        x[:, self.D:(self.D + self.C)] -> torch.Size([24, 64, 8, 22])
        x.unsqueeze(2)-> torch.Size([24, 64, 1, 8, 22])
        depth*x-> new_x: torch.Size([24, 64, 41, 8, 22])
        
        new_x: 24 x 64 x 41 x 8 x 22
        '''
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):  # 使用efficientnet提取特征
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))  #  x: 24 x 32 x 64 x 176
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x  # x: 24 x 320 x 4 x 11

        # 先对endpoints[4]进行上采样，然后将 endpoints[5]和endpoints[4] concat 在一起
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x  # x: 24 x 512 x 8 x 22

    # forward 返回带有深度信息的特征(调用get_depth_feat函数)
    def forward(self, x):
        '''
        depth: B*N x D x fH x fW(24 x 41 x 8 x 22)
        x: B*N x C x D x fH x fW(24 x 64 x 41 x 8 x 22)
        '''
        depth, x = self.get_depth_feat(x)
        return x

# 对BEV视图的特征进行编码（被LiftSplatShoot类中的初始化函数调用）
class BevEncode(nn.Module):
    def __init__(self, inC, outC):  # inC: 64  outC: 1
        super(BevEncode, self).__init__()

        # 使用resnet的前3个stage作为backbone
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(  # 2倍上采样->3x3卷积->1x1卷积
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):  # x: 4 x 64 x 200 x 200
        x = self.conv1(x)  # 下采样 /2， x: 4 x 64 x 100 x 100
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # x1: 4 x 64 x 100 x 100
        x = self.layer2(x1)  # 下采样 /2， x: 4 x 128 x 50 x 50
        x = self.layer3(x)   # 下采样 /2， x: 4 x 256 x 25 x 25

        x = self.up1(x, x1)  # 上采样 *4， 给x进行4倍上采样然后和x1 concat 在一起  x: 4 x 256 x 100 x 100
        x = self.up2(x)      # 上采样 *2， 2倍上采样->3x3卷积->1x1卷积  x: 4 x 1 x 200 x 200

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf          # 网格配置参数
        self.data_aug_conf = data_aug_conf  # 数据增强配置参数

        # 划分网格
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)  # [0.5,0.5,20] x,y,z方向上的网格间距
        self.bx = nn.Parameter(bx, requires_grad=False)  # [-49.5,-49.5,0] 三个轴上的起始坐标, # 第一个网格的中心坐标
        self.nx = nn.Parameter(nx, requires_grad=False)  # [200,200,1] 分别为x,y,z三个方向上格子的数量

        self.downsample = 16    # 下采样倍数
        self.camC = 64          # 图像特征维度
        self.frustum = self.create_frustum()  # frustum: DxfHxfWx3(41x8x22x3)
        self.D, _, _, _ = self.frustum.shape  # D: 41
        self.camencode = CamEncode(self.D, self.camC, self.downsample)  # D: 41 C:64 downsample:16
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    '''
    ------------------------------------------------------------------------------------------
    torch.linspace(start, end, steps=100, out=None, dtype=None,
                   layout=torch.strided, device=None, requires_grad=False)
    作用：返回一个一维的tensor（张量），这个张量包含了从start到end，分成steps个线段得到的向量
    常用参数：
      start：开始值
      end：结束值
      steps：分割的点数，默认是100
      dtype：返回值（张量）的数据类型
    ------------------------------------------------------------------------------------------
    '''

    '''
    create_frustum 为每一张图片生成一个棱台状（frustum）的点云 （被初始化函数调用）
    在图像平面上划分网格， 创造视锥
    1.根据图像像素坐标创建截锥体,截锥体的维度为DxfHxfWx3(fH = H/16, fW = W/16),在这里截锥体的维度是41x8x22x3。
    注意！！截锥体里保存的数据是什么，实际上是图像像素坐标在下采样之后的像素坐标。其实际意义就是在一个栅格中的图像坐标和深度(u,v,d)。
    
    看名字是建立视锥体，我觉得应该叫做create_sample_points，因为该部分代码是根据下采样倍率得到对应在输入图像上的采样点，
    并且在对应的采样点上补充设计的深度坐标，4到45米，41长度的深度。
    '''
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 原始图片大小  ogfH:128  ogfW:352
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 下采样16倍后图像大小  fH: 128/16=8  fW: 352/16=22
        '''
        ds： 在深度方向上划分网，每个特征点预测41个距离
        dbound: [4.0, 45.0, 1.0]   arange后-> [4.0,5.0,6.0,...,44.0]
        view后(相当于reshape操作)-> (41x1x1)    
        expand后（扩展张量中某维数据的尺寸）->  ds: DxfHxfW(41x8x22)
        '''
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape  # D: 41 表示深度方向上网格的数量

        '''
        xs: 在宽度方向上划分网格
        linspace 后(在[0,ogfW)区间内，均匀划分fW份)-> [0,16,32..336]  大小=fW(22)   
        view后-> 1x1xfW(1x1x22)
        expand后-> xs: DxfHxfW(41x8x22)
        '''
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # 在0到351上划分22个格子，视锥x坐标 xs: DxfHxfW(41x8x22)
        '''
        ys: 在高度方向上划分网格
        linspace 后(在[0,ogfH)区间内，均匀划分fH份)-> [0,16,32..112]  大小=fH(8)
        view 后-> 1xfHx1 (1x8x1)
        expand 后-> ys: DxfHxfW (41x8x22)
        '''
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # 在0到127上划分8个格子，视锥y坐标 ys: DxfHxfW(41x8x22)

        '''
        视锥点云，frustum: DxfHxfWx3
        D x H x W x 3  -> 41 x 8 x 22 x 3
        把xs,ys,ds堆叠到一起
        stack后-> frustum: DxfHxfWx3
        堆积起来形成网格坐标, frustum[d,h,w,0]就是(h,w)位置，深度为d的像素的宽度方向上的栅格坐标   
        # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标 
        '''
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)


    '''
        imgs:   图像数据
        rots: 相机坐标系到自车坐标系的旋转矩阵
        trans:  相机坐标系到自车坐标系的平移向量
        intrins:  相机内参
        post_rots:  数据增强的像素坐标旋转映射关系
        post_trans:   数据增强的像素坐标平移映射关系
        imgs: 4 x 5 x 3 x 128 x 352
        rots: 4 x 5 x 3 x 3]
        trans: 4 x 5 x 3
        intrins: 4 x 5 x 3 x 3
        post_rots: 4 x 5 x 3 x 3
        post_trans: 4 x 5 x 3
        binimgs: 4 x 1 x 200 x 200
    '''

    '''
    
    get_geometry 把在相机坐标系（ego frame）下的坐标 (x,y,z) 转换成自车坐标系下的点云坐标 (被get_voxels调用)
        
    锥点由图像坐标系向ego坐标系进行坐标转化
    结合相机的内外参数将截锥体中图像坐标先转化为相机坐标，再转化为车体坐标系的空间坐标(x,y,z)。
    geom_feats:6x4x41x8x22x3
    注意！！这里面的截锥体就生成了单目相机图像的像素在所有可能深度上的车体坐标系的三维空间位置。
    
    这里才是真正的建立视锥体，或者也可以叫做建立视锥点云，这里其实就是位置点，
    和图像特征一一对应，然后对引导图像特征填入BEV空间中。
    经过这一步操作就相当于在每个像素的每个离散深度值上都创造了一个点，之后再将这每个点通过相机内参和外参投影到自车坐标系下
    '''
    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 6(相机数目)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        # 但是数据增强是如何影响视锥的呢，为什么是减去，
        # 不应该是视锥也做同样的增强与图片对应起来嘛，而且他俩维度也不一样啊，怎么相减，是利用了广播机制吗 ？
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)  #减去数据增强时的平移变换
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))  #把图像转回去

        # cam_to_ego  相机坐标系转换成自车坐标系
        # 图像坐标系 -> 归一化相机坐标系 -> 相机坐标系 -> 车身坐标系
        # 但是自认为由于转换过程是线性的，所以反归一化是在图像坐标系完成的，然后再利用
        # 求完逆的内参投影回相机坐标系
        # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d), 我看了一下齐次坐标的定义，然后才明白，就是u，v,同时乘以d,然后在与d在拼接起来
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)  # 逆公式表达出x,y,z即可
        # 利用内外参将像素坐标转换到车体坐标下
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]^T

        # (bs, N, depth, H, W, 3)：其物理含义
        # 每个batch中的每个环视相机图像特征点，其在不同深度下位置对应
        # 在ego坐标系下的坐标

        return points  # B x N x D x H x W x 3 (4 x 6 x 41 x 8 x 22 x 3), 此时最后一维度代表的就是车坐标下的，x,y,z

    # get_cam_feats 调用camecode提取单张图像的特征 (被get_voxels调用)
    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 6  C: 3  imH: 128  imW: 352

        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x: 24 x 3 x 128 x 352
        x = self.camencode(x)  # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 8 x 22)
        # 此处将前两维度拆开时能保证顺序和合并前一样吗，不然我总怀疑最后图片输入的特征及深度分布如何映射到根据相机内外参数得到的视锥栅格里的
        # 将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 8 x 22)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 8 x 22 x 64)

        return x

    '''
    voxel_pooling 对voxel进行池化操作,调用了tools.py文件中定义的quicksum (被get_voxels调用)
    因为很多位置点会落到同一个grid里面，这里会做棱台累加和的操作。最后在一个pillar求平均，这一块代码比较复杂，但做得事情很简单，
    '''
    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 8 x 22 x 3)
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 8 x 22 x 64)

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 8  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime: 173184

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices
        # 将[-50,50] [-10 10]的范围平移到[0,100] [0,20]，计算栅格坐标并取整
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 (173184 x 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4(173184 x 4), geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~199  y: 0~199  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]  # x: 168648 x 64
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        # 从彼此相邻的相同体素中获取tensor
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort() # 按照rank排序，这样rank相近的点就在一起了
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x 1 x 200 x 200
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final  # final: 4 x 64 x 200 x 200

    '''
    1、get_voxels 先调用get_geometry把在相机坐标系（ego frame）下的坐标 (x,y,z) 转换成自车坐标系下的点云坐标；
    2、然后调用get_cam_feats提取单张图像特征，
    3、最后调用voxel_pooling 对体素特征进行汇聚。
    '''
    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # 确定点云中点的 (x,y,z) 位置（在小车坐标系中）。
        # 获取几何特征

        # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x 6 x 41 x 8 x 22 x 3)
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x 6 x 41 x 8 x 22 x 64)
        x = self.get_cam_feats(x)
        # x: 4 x 64 x 200 x 200
        x = self.voxel_pooling(geom, x)

        return x

    # forword 调用get_voxels把图像转换到BEV下，然后调用bevencode （初始化函数中定义，是BevEncode类的实例化）提取特征
    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        # x:[4,6,3,128,352]
        # rots: [4,6,3,3]
        # trans: [4,6,3]
        # intrins: [4,6,3,3]
        # post_rots: [4,6,3,3]
        # post_trans: [4,6,3]

        # 将图像转换到BEV下，x: B x C x 200 x 200 (4 x 64 x 200 x 200)
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        # 用resnet18提取特征  x: 4 x 1 x 200 x 200
        x = self.bevencode(x)
        return x

'''
    explore.py 中调用了compile_model函数。
    该函数构造了LiftSplatShoot 模型
'''
def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
