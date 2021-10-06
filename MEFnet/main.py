import numpy as np
import os
from PIL import Image
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import time
import json

from config import *
from dataset import *



# Parameters
init_c = 'previous'   # 'previous' / 'black'
class MEFO():

    def __init__(self):

        # ms parameters
        self.use_ms_ssim = True
        self.ms_scales = 3
        self.ms_weght = {1:0.0710, 2:0.4530, 3:0.4760}
     
        # ssim parameters
        self.d1 = 0.01
        self.d2 = 0.03
        self.C1 = (self.d1 * 255.0)**2
        self.C2 = (self.d2 * 255.0)**2

        # color image mean parameters
        self.sigma_g = 0.2
        self.sigma_l = 0.2
        self.channel = 3

        # cuda config
        self.use_cuda = torch.cuda.is_available()

        # create window        
        self.window_size = 8
        self.window = torch.ones(1, self.channel, self.window_size, self.window_size) / 192.0
        if self.use_cuda:
            self.window = self.window.float().cuda()
            
        
        if self.use_ms_ssim:
            self.max_iter = 3000
        else:
            self.max_iter = 1500
        
        # gradient-ascent algorithm parameters
        self.lambda_old  = torch.tensor(1.0)
        self.cvg_t = 2e-6
        self.beta_inverse = torch.tensor(120.0)
    
    # generate patch index and image seq parameters
    def generate_reference_patches(self, image_seq):
        
        denom_g = 2*self.sigma_g**2
        denom_l = 2*self.sigma_l**2
        a = torch.tensor(255.0)
        b = torch.tensor(0.5)
        num_images = len(image_seq)
        _, _, height, width = image_seq[0].shape
        lY = torch.zeros(num_images,1).cuda()
        mu_seq  = torch.zeros(height-self.window_size+1, width-self.window_size+1, num_images).cuda()
        mu_sq_seq  = torch.zeros(mu_seq.shape).cuda()
        sigma_sq_seq  = torch.zeros(mu_seq.shape).cuda()

        # for sigma mu_Y_seq should be used
        for k in range(num_images):
            mu_seq[:,:,k] = F.conv2d(image_seq[k], self.window)
            lY[k,0] = torch.mean(image_seq[k])
            mu_sq_seq[:,:,k] = mu_seq[:,:,k]**2
            sigma_sq_seq[:,:,k] =  F.conv2d(image_seq[k]**2, self.window) - mu_seq[:,:,k]**2
        sigma_sq, patch_index = torch.max(sigma_sq_seq, 2)
        mu = torch.zeros(height-self.window_size+1, width-self.window_size+1).cuda()
        LY = torch.zeros(mu_seq.shape).cuda()

        for k in range(num_images):
            LY[:,:,k] = torch.exp( - ((mu_seq[:,:,k] / a - b) ** 2) / denom_g - ( lY[k,0] / a - b )**2 / denom_l)
            mu += LY[:,:,k]*mu_seq[:,:,k]
        mu = mu / torch.sum(LY, 2)
        mu_sq = mu*mu

        self.patch_index = patch_index
        self.mu = mu
        self.mu_sq = mu_sq
        self.mu_seq = mu_seq
        self.mu_sq_seq = mu_sq_seq
        self.sigma_sq = sigma_sq
        self.sigma_sq_seq = sigma_sq_seq

    # compute cost and gradient
    def cost_ssimc(self, x, image_seq):

        x.requires_grad = True
        _, channel, height, width = x.shape
        num_images = len(image_seq)
        muX = torch.zeros(height-self.window_size+1, width-self.window_size+1).cuda()
        muX[:,:] = F.conv2d(x, self.window)
        muX_sq = muX**2
        sigmaX_sq = F.conv2d(x**2, self.window) - muX_sq
        cost_map = torch.zeros(self.mu_seq.shape).cuda()
        qmap = torch.zeros(muX.shape).cuda()
        sigmaXY = torch.zeros(self.mu_seq.shape).cuda()
        for k in range(num_images):
            sigmaXY[:,:,k] = F.conv2d(x*image_seq[k], self.window) - muX*self.mu_seq[:,:,k]
            cost_map[:,:,k] = (2 * muX * self.mu) * (2 * sigmaXY[:,:,k] + self.C2) / ((muX_sq  + self.mu_sq + self.C1) * (sigmaX_sq + self.sigma_sq_seq[:,:,k] + self.C2))
        
        for k in range(num_images):
            index = (self.patch_index == k)
            qmap = qmap + cost_map[:,:,k]*index.float()
        cost = torch.mean(qmap)
        sum_cost = torch.sum(qmap)
        sum_cost.backward()
        grad = x.grad
  
        return cost.detach(), grad

    # same as former but add scales
    def generate_reference_patches_ms(self, image_seq):
        
        denom_g = 2*self.sigma_g**2
        denom_l = 2*self.sigma_l**2
        num_images = len(image_seq)
        _, channel, height, width = image_seq[0].shape
                
        all_image_seq = {}
        all_mu_seq = {}
        all_mu_sq_seq = {}
        all_sigma_sq = {}
        all_sigma_sq_seq = {}
        all_lY = {}
        all_LY = {}
        all_mu = {}
        all_mu_sq = {}
        
        all_patch_index = {}

        for i in range(self.ms_scales):

            all_image_seq[i] = [] 
            all_lY[i] = torch.zeros(num_images,1).cuda()
            all_LY[i] =  torch.zeros((height // (2**i)) -self.window_size+1, (width // (2**i))  - self.window_size+1, num_images).cuda()

            all_mu[i] = torch.zeros((height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1).cuda()
            all_mu_seq[i] = torch.zeros((height // (2**i)) -self.window_size+1, (width // (2**i))  - self.window_size+1, num_images).cuda()
            all_mu_sq_seq[i] = torch.zeros((height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1, num_images).cuda()

            all_sigma_sq[i] = torch.zeros((height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1).cuda()
            all_sigma_sq_seq[i] = torch.zeros((height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1, num_images).cuda()
            all_patch_index[i] = torch.zeros((height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1).cuda()

            # Down-sampling 
            for k in range(num_images):
                all_image_seq[i].append(F.avg_pool2d(image_seq[k], (2**i, 2**i )))


            for k in range(num_images):
                all_mu_seq[i][:,:,k] = F.conv2d(all_image_seq[i][k], self.window)
                all_mu_sq_seq[i][:,:,k] = all_mu_seq[i][:,:,k]**2
                all_lY[i][k,0] = torch.mean(all_image_seq[i][k])
                all_sigma_sq_seq[i][:,:,k] = F.conv2d(all_image_seq[i][k]**2, self.window) - all_mu_sq_seq[i][:,:,k]

            all_sigma_sq[i], all_patch_index[i] = torch.max(all_sigma_sq_seq[i], 2)

            for k in range(num_images):
                all_LY[i][:,:,k] = torch.exp( - ((all_mu_seq[i][:,:,k] / 255.0 - 0.5) ** 2) / denom_g - ( all_lY[i][k,0] / 255.0 - 0.5 )**2 / denom_l)
                all_mu[i] += all_LY[i][:,:,k]*all_mu_seq[i][:,:,k]
            all_mu[i] = all_mu[i] / torch.sum(all_LY[i], 2)
            all_mu_sq[i] = all_mu[i]**2

        self.all_image_seq = all_image_seq
        self.all_patch_index = all_patch_index
        self.all_mu = all_mu
        self.all_mu_sq = all_mu_sq
        self.all_mu_seq = all_mu_seq
        self.all_mu_sq_seq = all_mu_sq_seq
        self.all_sigma_sq = all_sigma_sq
        self.all_sigma_sq_seq = all_sigma_sq_seq

    # same as former but add scales
    def cost_ms_ssim(self, x, image_seq):

        num_images = len(image_seq)
        x.requires_grad = True
        _, channel, height, width = x.shape

        all_muX = {}
        all_muX_sq = {}
        all_sigmaX_sq = {}

        all_sigmaXY = {}
        all_cost_map = {}
        all_qmap = {}
        qmap = torch.zeros(self.ms_scales,1)
        q_sum = torch.zeros(self.ms_scales,1)
         
        all_x = {}
        for i in range(self.ms_scales):

            all_muX[i] = torch.zeros( (height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1).cuda()
            all_muX_sq[i] = torch.zeros( (height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1).cuda()
            all_sigmaX_sq[i] = torch.zeros( (height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1).cuda()

            all_sigmaXY[i] = torch.zeros( (height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1, num_images).cuda()
            all_cost_map[i] = torch.zeros( (height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1, num_images).cuda()
            all_qmap[i] = torch.zeros( (height // (2**i))-self.window_size+1, (width // (2**i))-self.window_size+1).cuda()
            all_x[i] = []

            # Down-sampling 
            all_x[i].append( F.avg_pool2d(x, (2**i, 2**i )) )

            all_muX[i][:,:] = F.conv2d(all_x[i][0], self.window)
            all_muX_sq[i][:,:] = all_muX[i]**2
            all_sigmaX_sq[i][:,:] = F.conv2d(all_x[i][0]**2 , self.window)
            all_sigmaX_sq[i] -=  all_muX_sq[i]

            for k in range(num_images):             
                all_sigmaXY[i][:,:,k] = F.conv2d(all_x[i][0]*self.all_image_seq[i][k], self.window) - all_muX[i]*self.all_mu_seq[i][:,:,k]
                all_cost_map[i][:,:,k] =  ((2 * all_sigmaXY[i][:,:,k] + self.C2) / (all_sigmaX_sq[i] + self.all_sigma_sq_seq[i][:,:,k] + self.C2)) 
                if (i+1) == self.ms_scales:
                    all_cost_map[i][:,:,k] *= ((2 * all_muX[i] * self.all_mu[i])  / (all_muX_sq[i]  + self.all_mu_sq[i] + self.C1)) 
        
            for k in range(num_images):
                index = (self.all_patch_index[i] == k)
                all_qmap[i] = all_qmap[i] + all_cost_map[i][:,:,k]*index.float()

                 
            qmap[i,0] = torch.mean(all_qmap[i])**self.ms_weght[i+1]
            q_sum[i,0] = torch.sum(all_qmap[i])**self.ms_weght[i+1]


        cost = torch.prod(qmap)
        cost_sum = torch.prod(q_sum)
        cost_sum.backward()
        grad = x.grad

        return cost.detach(), grad

    def train(self, image_fused, image_seq, name):
            
        flow_num = []
        Cost = []
        x = image_fused
        cnt = 0
        for i in range(self.max_iter):

            if self.use_ms_ssim:
                cost, grad = self.cost_ms_ssim(x.detach(), image_seq)
            else:
                cost, grad = self.cost_ssimc(x.detach(), image_seq)

            if i != 0:
                if torch.abs(cost11-cost) < 2e-6:
                    cnt += 1
                if cnt > 100:
                    break

            cost11 = cost
            if i%50 == 0:
                print(name, cost, i)
                Cost.append(float(np.asarray(cost.cpu())))
            #print('cost', cost)
            if i== 0:
                Cost1 = float(np.asarray(cost.cpu()))

            lamda_new = ( 1 + torch.sqrt(1 + 4*self.lambda_old**2 )) / 2.0
            gamma = (1 - self.lambda_old) / lamda_new

            y_new = x + self.beta_inverse*grad
            x_new = (1-gamma)*y_new + gamma*x

            overflow_ind = x_new > 255.0
            x_new[overflow_ind] = 255.0
            underflow_ind = x_new < 0.0
            x_new[underflow_ind] = 0.0

            flow_num.append(torch.sum(overflow_ind) + torch.sum(underflow_ind))

            x = x_new

        output_image = np.transpose(np.squeeze(np.asarray(x.cpu())), (1,2,0))
        output_image = output_image / 255.0 
    
        return Cost, output_image, Cost1
                 
def main():

    mefo = MEFO()
    use_ms_ssim = True
    verify = False
    init_cost = []
    for src_fd in os.listdir(src_path):
        meta = {}
        save_path = save_dir + '/' + src_fd
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(len(all_fused[src_fd])):

            if init_c == 'previous':
                name = mapping_dict[src_fd][i]
            elif init_c == 'black':
                name = src_fd
            else:
                1/0

            # skip already optimized images
            if name.split('.')[0].split('-')[-1] == 'optimized':
                continue

            image_seq = all_seq[src_fd]
            image_fused = all_fused[src_fd][i]

            # use tower image for verification (compare with Matlab original implementation)
            if verify:
                image_seq = tower_seq
                image_fused = tower_fused
            
            # choose use ms-ssim or ssim as metric
            if use_ms_ssim:
                mefo.generate_reference_patches_ms(image_seq)
            else:
                mefo.generate_reference_patches(image_seq)
            
            # timing
            t1 = time.time()

            # training
            Cost, output_image, Cost1 = mefo.train(image_fused, image_seq, name) 
            init_cost.append(Cost1)   
            meta[name] = Cost
            
            if use_ms_ssim:
                tag = '_ms_ssim_opt'
            else:
                tag = '_ssim_opt'

            plt.imsave(save_path + '/' + name + tag +'.png', output_image)
            plt.imsave(save_dir + '/' + name + tag +'.png', output_image)
            t2 = time.time()
            print('time for optimizing', src_fd, i, t2-t1)
        
        f = open( save_path + '/' + 'all_cost' + '_' + init_c + '.json', 'w')
        json.dump(meta, f)
        f.close()

        f = open( save_dir + '/' + 'init_cost' + '_' + init_c + '.json', 'w')
        json.dump(init_cost, f)
        f.close()

if __name__ == "__main__":

    # Verification purpose
    tower_path = cfg.tower_path
    tower_fused_p = cfg.tower_fused_p

    # make save dir
    save_dir = cfg.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get tower initial image and input sequence (if needed)
    tower_seq = []
    for item in os.listdir(tower_path):
        image_tensor = np.asarray(Image.open(tower_path + '/' + item))
    tower_fused = np.asarray(Image.open(tower_fused_p))

    # get datasets 
    mef_dataset = 

    main()


