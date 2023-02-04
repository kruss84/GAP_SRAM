import sys
sys.path.append(".")
from train_net import *

cfg=Config('volleyball')

cfg.use_multi_gpu=False
cfg.device_list="0,1"
cfg.training_stage=1
cfg.stage1_model_path='data/volleyball/videos/STAGE1_MODEL.pth'
#cfg.stage1_model_path='result/stage1_82.pth'
#cfg.stage2_model_path='/home/junwen/opengit/ARG_group/ARG_source/result/[Volleyball_stage2_stage2]<2019-08-05_16-20-34>/stage2_epoch6_83.82%.pth'
#cfg.resume=True
cfg.train_backbone=True
cfg.lr_plan={}

cfg.batch_size=4
cfg.test_batch_size=2
cfg.num_frames=20
cfg.train_learning_rate=1e-4
# cfg.test_before_train=True

cfg.exp_note='Volleyball_stage1'
train_net(cfg)
