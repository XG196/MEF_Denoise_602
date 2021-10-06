from PIL import Image
import numpy as np
import os

from config import *




class MEFSSIMC_Database():

    def __init__(self, src_path=cfg.src_path_0, pre_path=cfg.pre_path):

        self.init_img = []
        self.input_seq = []
        
        if cfg.exp_id == 0:
            src_path=cfg.src_path_0
        elif cfg.exp_id == 1:
            src_path=cfg.src_path_1
        else:
            raise NotImplementedError(" experiment not implemented !")

        for img in os.listdir(pre_path):
            fd_name = img.split('_')[0]
            src_path = src_path + '/' + fd_name
            pre_p = pre_path + '/' + img

            if cfg.exp_id == 0:
                img1  = np.asarray(Image.open(pre_p))
                self.init_img.append(np.ones(img1.shape))
            elif cfg.exp_id == 1:
                img1 = np.asarray(Image.open(pre_p))
                self.init_img.append(img1)
            else:
                raise NotImplementedError(" experiment not implemented !")


            img3_list = []
            for img2 in os.listdir(src_path):
                src_p = src_path + '/' + img2
                img3 = np.asarray(Image.open(src_p))
                img3_list.append(img3)
            self.input_seq.append(img3_list)
    
    



