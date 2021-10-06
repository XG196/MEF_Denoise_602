from easydict import EasyDict as edict


__C = edict()


# cfg is treated as a global variable. 
# Need add "from config import cfg" for each file to use cfg.
cfg = __C

__C.exp_id = 1
__C.save_dir = '/home/x227guo/workspace/MEFNet/results/results'


# Dataset path
__C.src_path_0 = '/home/x227guo/workspace/MEFNet/MEFOpt_Database/database_release/exp_5/source_images_n'
__C.src_path_1 = '/home/x227guo/workspace/MEFNet/MEFOpt_Database/database_release/source_images'
__C.pre_path = '/home/x227guo/workspace/MEFNet/MEFOpt_Database/database_release/exp_5/fused_images_n'
__C.tower_path = '/home/x227guo/workspace/MEFNet/MEFOpt/images/Tower'
__C.tower_fused_p = '/home/x227guo/workspace/MEFNet/MEFOpt/images/Tower_Mertens07.png'

__C.use_cuda = False
