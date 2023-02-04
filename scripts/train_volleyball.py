import sys
sys.path.append(".")
from train_net import *

cfg=Config('volleyball')

cfg.dataset_name = 'volleyball'

cfg.use_multi_gpu=False
cfg.device_list="0,1"
cfg.training_stage=2
cfg.stage1_model_path='data/volleyball/videos/stage1_epoch68_87.81%.pth'
#cfg.stage1_model_path='result/stage1_82.pth'
#cfg.stage2_model_path='/home/junwen/opengit/ARG_group/ARG_source/result/[Volleyball_stage2_stage2]<2019-08-05_16-20-34>/stage2_epoch6_83.82%.pth'
#cfg.resume=True
cfg.train_backbone=False
cfg.lr_plan={}

cfg.batch_size=4
cfg.test_batch_size=2
cfg.num_frames=20
cfg.train_learning_rate=1e-4
# cfg.test_before_train=True

cfg.exp_note='Volleyball_stage2'
train_net(cfg)
