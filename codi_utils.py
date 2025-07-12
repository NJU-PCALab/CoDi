# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from diffusers import DDIMScheduler
from codi_unet_sdxl import CoDiSDXLUNet2DConditionModel
from codi_pipeline import CoDiExtendAttnSDXLPipeline
from utils.general_utils import *

class OptimalTransport:
    def __init__(self):
        self.fea={}
    
    def set_fea(self,resolution,fea):
        self.fea[resolution]=fea

    def set_subject_mask(self,subject_mask):
        self.subject_mask=subject_mask
    
    def set_attn_map(self,attn_map):
        self.attn_map=attn_map
    
    def compute_cost_and_weights(self,X, Y):
            """
            X: shape [M, D] )
            Y: shape [N, D] )
            Returns:
                cost_matrix: shape [M, N]
                s_x: normalized weights for X, shape [M]
                s_y: normalized weights for Y, shape [N]
            """
            
            y_mean = Y.mean(dim=0) 
            x_mean = X.mean(dim=0)
            
            s_x = torch.clamp_min(X @ y_mean, 0) 
            s_y = torch.clamp_min(Y @ x_mean, 0) 
            
            s_x = s_x * (X.shape[0] / s_x.sum())    
            s_y = s_y * (X.shape[0] / s_y.sum())  

            s_x = s_x.view(-1, 1) 
            s_y = s_y.view(-1, 1)
            
            X_norm = F.normalize(X, p=2, dim=1) 
            Y_norm = F.normalize(Y, p=2, dim=1) 
            
            cos_sim = X_norm @ Y_norm.T   
            
            cost_matrix = 1 - cos_sim    

            return cost_matrix, s_x, s_y
    
    def compute_OT_plan(self,cost_matrix, weight1, weight2):
        weight1=torch.softmax(weight1,dim=-1)
        weight2=torch.softmax(weight2,dim=-1)
        
        cost_matrix = cost_matrix.cpu().numpy().astype(np.float32)
        weight1 = weight1.cpu().numpy().astype(np.float32)
        weight2 = weight2.cpu().numpy().astype(np.float32)
        
        cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
        return cost, flow
    def get_OT_plan(self):
        
        OT_plan = {}
        cost_dict ={}
        for res in self.fea.keys():
            
            fea = self.fea[res]     
            masks = self.subject_mask[res]  
            B, N, D = fea.shape

            src_fea = fea[0]              
            src_mask = masks[0]            
            X = src_fea[src_mask]  
            X=X.float()

            flow_list = []
            cost_list = []
            for i in range(1, B):
                tgt_fea = fea[i]            
                tgt_mask = masks[i]     
                Y = tgt_fea[tgt_mask]      
                
                Y=Y.float()
                cost_matrix, s_x, s_y=self.compute_cost_and_weights(X,Y)
                
                s_x=self.attn_map[res][0][src_mask]
                s_y=self.attn_map[res][i][tgt_mask]
                
                cost, flow=self.compute_OT_plan(cost_matrix, s_x, s_y)
                
                flowT=flow.T
                if isinstance(flowT, np.ndarray):
                    flowT = torch.from_numpy(flowT).to(device=X.device).to(torch.float16)
                flowT=flowT/flowT.sum(dim=1, keepdim=True)
                flow_list.append(flowT)
                
                cost_list.append(1-cost_matrix.T)

            OT_plan[res]=flow_list
            cost_dict[res]=cost_list
        
        return OT_plan,cost_dict
    

def load_pipeline(gpu_id=0):
    float_type = torch.float16
    sd_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    unet = CoDiSDXLUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", torch_dtype=float_type) 
    scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")

    story_pipeline = CoDiExtendAttnSDXLPipeline.from_pretrained(
        sd_id, unet=unet, torch_dtype=float_type, variant="fp16", use_safetensors=True, scheduler=scheduler
    ).to(device)
    story_pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2) 
    
    return story_pipeline

def create_anchor_mapping(bsz, anchor_indices=[0]):
    anchor_mapping = torch.eye(bsz, dtype=torch.bool)
    for anchor_idx in anchor_indices:
        anchor_mapping[:, anchor_idx] = True

    return anchor_mapping

def create_token_indices(prompts, batch_size, concept_token, tokenizer):
    if isinstance(concept_token, str):
        if " " in concept_token:
            concept_token=concept_token.split(" ")
        else:
            concept_token = [concept_token]

    concept_token_id = [tokenizer.encode(x, add_special_tokens=False)[0] for x in concept_token] 
    
    tokens = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')['input_ids'] 
    token_indices = torch.full((len(concept_token), batch_size), -1, dtype=torch.int64) 
    for i, token_id in enumerate(concept_token_id): 
        batch_loc, token_loc = torch.where(tokens == token_id)
        token_indices[i, batch_loc] = token_loc
    
    return token_indices


def create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type):
    # if seed is int
    if isinstance(seed, int):
        g = torch.Generator('cuda').manual_seed(seed)
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = randn_tensor(shape, generator=g, device=device, dtype=float_type)
    elif isinstance(seed, list):
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = torch.empty(shape, device=device, dtype=float_type)
        for i, seed_i in enumerate(seed):
            g = torch.Generator('cuda').manual_seed(seed_i)
            curr_latent = randn_tensor(shape, generator=g, device=device, dtype=float_type)
            latents[i] = curr_latent[i]
    
    if same_latent:
        latents = latents[:1].repeat(batch_size, 1, 1, 1)

    return latents, g
