"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from fire import Fire

import src


'''
python fire使用指南: https://blog.csdn.net/qq_17550379/article/details/79943740

fire.Fire()： 通过使用字典，我们可以有选择性地将一些函数暴露给命令行。
然后可以通过命令行传参的方式使用指定函数。
如： python main.py eval_model_iou      
运行src.explore.eval_model_iou(),
然后程序就开始执行src/explore.py文件下的eval_model_iou 函数。
'''


if __name__ == '__main__':
    Fire({
        'lidar_check': src.explore.lidar_check,
        'cumsum_check': src.explore.cumsum_check,

        'train': src.train.train,
        'eval_model_iou': src.explore.eval_model_iou,
        'viz_model_preds': src.explore.viz_model_preds,
    })

    # src.train.train(version='mini', dataroot='./data/nuscenes', logdir='./runs', gpuid=0, bsz=2)

    # src.explore.viz_model_preds(version='mini', modelf='./weights/pre_model.pt', dataroot='./data/nuscenes',
    #                             map_folder='./data/nuscenes/mini')