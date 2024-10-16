import os
from re import L
import time
import functools
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ApertureCamera(nn.Module):
    def __init__(self, data_dict, aperture_sample_rate = 3, aperture_sample_scale=5, shared_intrinsic = True, shared_settings=False):
        super(ApertureCamera, self).__init__()        
        self.data_dict = data_dict
        
        self.shared_intrinsic = shared_intrinsic
        self.shared_settings = shared_settings

        # get extrinsics -----------------------------------
        #self.extrinsics = nn.Parameter(torch.tensor(data_dict['poses']))
        #self.register_buffer('extrinsics', torch.tensor(data_dict['poses']))
        self.register_buffer('extrinsics', data_dict['poses'])
        self.extrinsics_residual = nn.Parameter(torch.zeros_like(self.extrinsics))
        self.n_imgs = len(self.extrinsics)
        #'near', 'far', 'near_clip', 'i_train', 'i_val', 'i_test', 'render_poses', 'images', 'irregular_shape',        

        # get intrinsics  ------------------------------------
        if shared_intrinsic: # (1x4)
            self.register_buffer('Hs', torch.from_numpy(data_dict['HW'][0:1, 0]))
            self.register_buffer('Ws', torch.from_numpy(data_dict['HW'][0:1, 1]))
            #self.intrinsics = nn.Parameter(torch.tensor([[data_dict['Ks'][0,0,0], data_dict['Ks'][0,1,1], data_dict['Ks'][0,0,2], data_dict['Ks'][0,1,2]]]))
            self.register_buffer('intrinsics', torch.tensor([[data_dict['Ks'][0,0,0], data_dict['Ks'][0,1,1], data_dict['Ks'][0,0,2], data_dict['Ks'][0,1,2]]]))
            self.intrinsics_residual = nn.Parameter(torch.zeros_like(self.intrinsics))
            self.register_buffer('distortion', torch.tensor([data_dict['intrinsics'][-4:]]))
        else: # (n_imgsx4)
            raise RuntimeError('unhandeld shared_intrinsic=False case')
            self.register_buffer('Hs', torch.from_numpy(data_dict['HW'][:, 0]))
            self.register_buffer('Ws', torch.from_numpy(data_dict['HW'][:, 1]))
            self.intrinsics = nn.Parameter(torch.tensor(np.stack([data_dict['Ks'][:,0,0], data_dict['Ks'][:,1,1], data_dict['Ks'][:,0,2], data_dict['Ks'][:,1,2]], axis=1)))
            
        
        # get lens information --------------------------
        if not ('lens_info_list' in data_dict.keys()):
            self.is_pinhole = True
            self.col2pix = None
            self.col2pix_residual = None
            self.lensFp = None
            self.lensDp = None
            self.mm2pix = None
            self.expT = None
            self.expS = None
            self.apt_sample_rates = 1
            return

        self.is_pinhole = False
        self.register_buffer('col2pix', torch.tensor(data_dict['col2X'][0]))
        self.col2pix_residual = nn.Parameter(torch.zeros_like(self.col2pix)) 

        if shared_settings:
            self.register_buffer('lensFp', torch.tensor([[data_dict['lens_info_list'][0]['lensFXp'], data_dict['lens_info_list'][0]['lensFYp']]]))
            self.register_buffer('lensDp', torch.tensor([[data_dict['lens_info_list'][0]['lensDXp'], data_dict['lens_info_list'][0]['lensDYp']]]))            
            self.register_buffer('mm2pix', torch.tensor([[data_dict['lens_info_list'][0]['mm2pixX'], data_dict['lens_info_list'][0]['mm2pixY']]]))            
            self.expT = nn.Parameter(torch.tensor([data_dict['lens_info_list'][0]['expT']])) # exposure time
            self.expS = nn.Parameter(torch.ones((1,3))) # exposure scale for each channel

            # self.lensFp_residual = nn.Parameter(torch.zeros_like(self.lensFp))
            
        else:
            self.register_buffer('lensFp', torch.tensor([[lens_info['lensFXp'], lens_info['lensFYp']] for lens_info in data_dict['lens_info_list']]))
            self.register_buffer('lensDp', torch.tensor([[lens_info['lensDXp'], lens_info['lensDYp']] for lens_info in data_dict['lens_info_list']]))
            self.register_buffer('mm2pix', torch.tensor([[lens_info['mm2pixX'], lens_info['mm2pixY']] for lens_info in data_dict['lens_info_list']]))
            self.expT = nn.Parameter(torch.tensor([[lens_info['expT']] for lens_info in data_dict['lens_info_list']])) # exposure time
            self.expS = nn.Parameter(torch.ones((self.n_imgs,3))) # exposure scale for each channel

            # self.lensFp_residual = nn.Parameter(torch.zeros_like(self.lensFp))

        #self.register_buffer('apt_sample_rates', 22//self.get_Fnumbers())                
        # self.apt_sample_rates = torch.ceil(22/self.get_Fnumbers())
        # self.register_buffer('apt_sample_rates', torch.ceil(22/self.get_Fnumbers())+2)
                
        self.register_buffer('apt_sample_rates', torch.full_like(self.get_Fnumbers(), aperture_sample_rate))                

        # self.register_buffer('apt_sample_rates', torch.full_like(self.get_Fnumbers(), 7))        
        # self.register_buffer('apt_sample_rates', torch.full_like(self.get_Fnumbers(), 5))
        # self.register_buffer('apt_sample_rates', torch.full_like(self.get_Fnumbers(), 3))        
        # self.register_buffer('apt_sample_rates', torch.full_like(self.get_Fnumbers(), 1))        
        #self.apt_sample_rates += (self.apt_sample_rates % 2 == 0).int()  # make it odd number
        self.apt_sample_rates = torch.clamp_max(self.apt_sample_rates[:, 0].int(), 9)
        #self.apt_sample_rates = torch.clamp_max(self.apt_sample_rates[:, 0].int(), 1)

        self.register_buffer('apt_sample_rates_eval', torch.floor(35/self.get_Fnumbers())) # original with max 9   
        self.apt_sample_rates_eval += (self.apt_sample_rates_eval % 2 == 0).int()  # make it odd number
        self.apt_sample_rates_eval = torch.clamp_max(self.apt_sample_rates_eval[:, 0].int(), 9)

        # if (aperture_sample_rate*aperture_sample_scale) % 2 == 0:
        #     aperture_sample_scale += 1/aperture_sample_rate
        self.register_buffer('apt_sample_scales', torch.full_like(self.get_Fnumbers(), aperture_sample_scale))        
        # self.apt_sample_scales = torch.clamp_max(self.apt_sample_scales[:, 0].int(), 9)
        self.apt_sample_scales = torch.clamp_max(self.apt_sample_scales[:, 0].int(), 21)

        print(f'Apt Sample Rate Min: {self.apt_sample_rates.min()}')
        print(f'Apt Sample Rate Max: {self.apt_sample_rates.max()}')

    def get_kwargs(self):
        return {'data_dict':self.data_dict}

    def intrinsics2Ks(self):
        Ks = torch.zeros((self.n_imgs, 3, 3))
        intrinsics = self.intrinsics + self.intrinsics * self.intrinsics_residual
        lensFps = self.get_lensFps()
        
        if intrinsics.shape[0] != self.n_imgs:
            intrinsics = intrinsics.expand(self.n_imgs, -1)
        else:
            intrinsics = intrinsics
        
        for i, (intrinsic, lensFp) in enumerate(zip(intrinsics, lensFps)):
            # Ks[i][0,0] = intrinsic[0]
            # Ks[i][1,1] = intrinsic[1]
            # Ks[i][0,2] = intrinsic[2]
            # Ks[i][1,2] = intrinsic[3]
            # Ks[i][0,0] = torch.clamp_min(intrinsic[0], lensFp[0])
            # Ks[i][1,1] = torch.clamp_min(intrinsic[1], lensFp[1])
            Ks[i][0,0] = torch.clamp_min(intrinsic[0], lensFp[0].item())
            Ks[i][1,1] = torch.clamp_min(intrinsic[1], lensFp[1].item())
            Ks[i][0,2] = intrinsic[2]
            Ks[i][1,2] = intrinsic[3]
            Ks[i][2,2] = 1.0
        
        return Ks
    
    def get_col2pix(self):
        col2pix = torch.clamp_min(self.col2pix + (self.col2pix_residual * self.col2pix), 0)
        # col2pix = torch.clamp_min(self.col2pix + self.col2pix_residual, 0)
        # col2pix = torch.clamp_min(self.col2pix + self.col2pix_residual*self.get_Ws()[0].item(), 0)
        return col2pix

    def get_col2mms(self):
        col2pix = self.get_col2pix()
        return col2pix / self.mm2pix

    def get_Hs(self, idx=None):
        if self.Hs.shape[0] == 1:
            Hs = self.Hs.expand(self.n_imgs, -1)

        if idx is not None:
            return Hs[idx]        
        else:
            return Hs
    
    def get_Ws(self, idx=None):
        if self.Ws.shape[0] == 1:
            Ws = self.Ws.expand(self.n_imgs, -1)
        if idx is not None:
            return Ws[idx]        
        else:
            return Ws
    
    def get_Ks(self, idx=None):
        Ks = self.intrinsics2Ks()
        if idx is not None:            
            return Ks[idx]
        else:
            return Ks
    
    def get_RTs(self, idx=None):
        ext_messy = self.extrinsics + self.extrinsics_residual
        #ext_messy = self.extrinsics.clone()
        b1 = ext_messy[:, :, 0]/ext_messy[:, :, 0].norm(dim=1, keepdim=True)
        b2_ = ext_messy[:, :, 1] - (ext_messy[:, :, 1]*b1).sum(dim=1,keepdim=True)*b1
        b2 = b2_/b2_.norm(dim=1, keepdim=True)
        b3 = torch.cross(b1, b2)
        
        ext_clean = torch.stack([b1, b2, b3], dim=2)
        ext_clean = torch.cat([ext_clean, ext_messy[:, :, 3:4]], dim=2)
        if idx is not None:
            # return self.extrinsics[idx]
            return ext_clean[idx]
        else:
            #return self.extrinsics
            return ext_clean

    def get_lensFps(self, idx=None):
        if self.lensFp.shape[0] == 1:
            lensFps = self.lensFp.expand(self.n_imgs, -1)
        else:
            lensFps = self.lensFp

        # lensFps_final = torch.clamp_min(lensFps + self.lensFp_residual, 0)
        # # lensFps_final = torch.clamp_min(lensFps + self.lensFp_residual*self.get_Ws()[0].item(), 0)
        lensFps_final = lensFps
        
        if idx is not None:
            return lensFps_final[idx]
        else:
            return lensFps_final
    
    def get_lensDps(self, idx=None):
        if self.lensDp.shape[0] == 1:
            lensDps = self.lensDp.expand(self.n_imgs, -1)
        else:
            lensDps = self.lensDp
        if idx is not None:
            return lensDps[idx]
        else:
            return lensDps
        
    def get_total_n_piexls(self, subset=None):
        Ws = self.get_Ws(subset)
        Hs = self.get_Hs(subset)
        return torch.sum(Hs*Ws).item()        

    def index2position(self, indices, subset=None):
        Ws = self.get_Ws(subset)
        Hs = self.get_Hs(subset)
        indices = indices.to(Ws.device)
        if self.shared_intrinsic:
            idx_img = indices//(Ws[0]*Hs[0])
            idx_pixel = indices%(Ws[0]*Hs[0])
            x = idx_pixel % Ws[0]
            y = idx_pixel // Ws[0]
            return torch.stack([idx_img, x, y], dim=1)
        else:
            cumpix = torch.cumsum( Ws * Hs, dim=0 )

            positions = []
            for index in indices:
                idx_img = (torch.nonzero(cumpix >= index+1)[0,0]).item()
                if idx_img == 0:
                    idx_pixel = index 
                else:
                    idx_pixel = (index - cumpix[idx_img-1]).item()
                    if idx_pixel < 0:
                        raise RuntimeError('index shoule be greater than -1')
                x = idx_pixel % Ws[idx_img].item()
                y = idx_pixel // Ws[idx_img].item()
                positions.append(torch.tensor([idx_img, x, y]))
            return torch.stack(positions, dim=0)

    def get_Fnumbers(self):
        lensFps = self.get_lensFps()
        lensDps = self.get_lensDps()
        return lensFps/lensDps
    
    def get_apt_sample_rates(self, idx=None):
        with_scale = not (torch.all(self.apt_sample_scales==1))
        if self.training:
            if idx is None:
                return self.apt_sample_rates
            return self.apt_sample_rates[idx]
            
        elif with_scale:    
            final_sample_rates = self.apt_sample_rates.clone()
            final_sample_rates = torch.clamp_max( self.apt_sample_rates_eval, self.apt_sample_rates)
            final_sample_rates = final_sample_rates.masked_fill((self.get_Fnumbers()==22)[:,0],1)
            if idx is None:
                return final_sample_rates
            return final_sample_rates[idx]

        else:            
            # self.register_buffer('apt_sample_rates_eval', torch.floor(35/self.get_Fnumbers())) # original with max 9            
            # self.apt_sample_rates_eval += (self.apt_sample_rates_eval % 2 == 0).int()  # make it odd number
            
            # warn. this number of samples is different from the ones that used for video synthesis.
            # please check render.py > do_rendering > apt_sample_rates
            #self.apt_sample_rates_eval = torch.clamp_max(self.apt_sample_rates_eval[:, 0].int(), 12)
            #self.apt_sample_rates_eval = self.apt_sample_rates_eval[:, 0].int()

            print(f'Fnumber Min: {self.get_Fnumbers().min()}')
            print(f'Fnumber Max: {self.get_Fnumbers().max()}')
            if idx is None:
                print(f'Apt Sample Rate Min: {self.apt_sample_rates_eval.min()}')
                print(f'Apt Sample Rate Max: {self.apt_sample_rates_eval.max()}')
            else:
                print(f'Apt Sample Rate Min: {self.apt_sample_rates_eval[idx].min()}')
                print(f'Apt Sample Rate Max: {self.apt_sample_rates_eval[idx].max()}')
        
            if idx is None:
                return self.apt_sample_rates_eval
            return self.apt_sample_rates_eval[idx]
            
    def get_apt_sample_scales(self, idx):
        with_scale = not (torch.all(self.apt_sample_scales==1))
        # if self.training or (not with_scale):
        if with_scale:
            apt_sample_rates = self.get_apt_sample_rates(None)
            # self.register_buffer('apt_sample_rates_eval', torch.max( torch.floor(22/self.get_Fnumbers()), 1))
            #apt_sample_scales_eval = torch.clamp_min( torch.floor(22/self.get_Fnumbers()), 1)
            # apt_sample_scales_eval = torch.clamp_min( torch.ceil(22/self.get_Fnumbers()), 1)
            
            apt_sample_scales_eval = self.compute_apt_sample_scales(self.get_Fnumbers())

            # apt_sample_scales_eval = torch.clamp_max(apt_sample_scales_eval[:, 0].int(), 31) #9)
            # apt_sample_scales_eval = torch.clamp_max(apt_sample_scales_eval[:, 0].int(), 25//apt_sample_rates) #9)
            # apt_sample_scales_eval += (apt_sample_scales_eval % 2 == 0).int()  # make it odd number            
            # apt_sample_scales_eval += (1/apt_sample_rates)*((apt_sample_rates*apt_sample_scales_eval) % 2 == 0)            

            print(f'Fnumber Min: {self.get_Fnumbers().min()}')
            print(f'Fnumber Max: {self.get_Fnumbers().max()}')
            if idx is None:
                print(f'Apt Sample Rate Min: {apt_sample_rates.min()}')
                print(f'Apt Sample Rate Max: {apt_sample_rates.max()}')
                print(f'Apt Sample Scale Min: {apt_sample_scales_eval.min()}')
                print(f'Apt Sample Scale Max: {apt_sample_scales_eval.max()}')
                return apt_sample_scales_eval
            else:
                print(f'Apt Sample Rate Min: {apt_sample_rates[idx].min()}')
                print(f'Apt Sample Rate Max: {apt_sample_rates[idx].max()}')
                print(f'Apt Sample Scale Min: {apt_sample_scales_eval[idx].min()}')
                print(f'Apt Sample Scale Max: {apt_sample_scales_eval[idx].max()}')
                return apt_sample_scales_eval[idx]
        else:
            if idx is None:
                return self.apt_sample_scales # init 1
            return self.apt_sample_scales[idx]
        
    def compute_apt_sample_scales(self, f_numbers):
        apt_sample_scales_eval =  torch.floor(35/f_numbers)
        #apt_sample_scales_eval =  torch.floor(43/f_numbers)
        apt_sample_scales_eval += (apt_sample_scales_eval % 2 == 0).int()  # make it odd number
        apt_sample_scales_eval = torch.clamp_max(apt_sample_scales_eval[:, 0].int(), 19)
        apt_sample_scales_eval = torch.clamp_min(apt_sample_scales_eval, 3)
        return apt_sample_scales_eval

    def get_pix2mm(self):
        return 1/self.mm2pix
    
    def get_mm2pix(self):
        return self.mm2pix

    def get_distortions(self, idx=None):
        if self.distortion.shape[0] == 1:
            distortion = self.distortion.expand(self.n_imgs, -1)
        else:
            distortion = self.distortion
        if idx is not None:
            return distortion[idx]
        else:
            return distortion

    # loss -------------------------------------
    def get_extrinsics_loss(self):
        loss = self.get_RTs() - self.extrinsics
        return torch.mean(loss * loss)
        # diverge
        #return (self.extrinsics_residual*self.extrinsics_residual).sum()

    def get_col_loss(self):
        #col2pix = self.get_col2pix() # torch.Size([2])
        #pix2mm = self.get_pix2mm() # torch.Size([50, 2])
        #diff = torch.abs(col2pix[0] * pix2mm[:, 0] - col2pix[1] * pix2mm[:, 1])
        #return torch.mean(diff * diff, dim=0)
        return torch.mean(self.col2pix_residual**2)

    # def get_lensfp_loss(self):
    #     return torch.mean(self.lensFp_residual**2)
        
    def get_intrinsics_loss(self):
        #Ks = self.get_Ks()
        #pix2mm = self.get_pix2mm()
        #loss = Ks[:,0,0]*pix2mm[:,0] - Ks[:,1,1]*pix2mm[:,1]
        #return torch.mean(loss * loss, dim=0)
        return torch.mean(self.intrinsics_residual**2)

    

