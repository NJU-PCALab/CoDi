# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling ConsiStory or otherwise documented as NVIDIA-proprietary
# are not a contribution and subject to the license under the LICENSE file located at the root directory.

from diffusers.utils import USE_PEFT_BACKEND
from typing import Callable, Optional
import torch
from diffusers.models.attention_processor import Attention

import torch.nn.functional as F
from diffusers.utils.import_utils import is_xformers_available
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class CoDiAttnStoreProcessor: 
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, 
        hidden_states, 
        encoder_hidden_states=None, 
        record_attention=True,
        curr_iter=None,
        subject_masks=None,
        OT_plan=None,
        transition_point=None,
        vanilla=False,
        **kwargs):
        
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key)
        
        if record_attention:
            if curr_iter>=49:
                self.attnstore(attention_probs, is_cross, self.place_in_unet, attn.heads)
        
        hidden_states = torch.bmm(attention_probs, value)
        
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if not vanilla and curr_iter<transition_point:
            resolution=int(hidden_states.shape[1]**0.5)
            bs=subject_masks[resolution].shape[0]
            for i in range(1,bs):
                hidden_states[bs+i:bs+i+1,subject_masks[resolution][i]] = torch.einsum("cm,bmd->bcd", OT_plan[resolution][i-1], hidden_states[bs:bs+1,subject_masks[resolution][0]])

        return hidden_states

class CoDiExtendedAttnXFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, place_in_unet, attnstore, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

        self.place_in_unet = place_in_unet
        self.curr_unet_part = self.place_in_unet.split('_')[0]
        self.attnstore = attnstore

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        curr_iter=None,
        subject_masks=None,
        OT_plan=None,
        transition_point=None,
        optimalTransport=None,
        vanilla=False,
        identity_top_alpha_masks=None,
        **kwargs
    ) -> torch.FloatTensor:
        
        residual = hidden_states 
        
        args = () if USE_PEFT_BACKEND else (scale,)
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, wh, channel = hidden_states.shape
            height = width = int(wh ** 0.5)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query).contiguous()
        ori_dtype=query.dtype
        
        if not vanilla and curr_iter>transition_point:
            half_bs=batch_size//2
            ex_out = torch.empty_like(query)
            resolution=int(hidden_states.shape[1]**0.5) 
            
            identity_top_alpha_mask=identity_top_alpha_masks[resolution]
            
            for i in range(batch_size):
                start_idx = i * attn.heads
                end_idx = start_idx + attn.heads
                curr_q = query[start_idx:end_idx]
                curr_k=key[i:i+1]
                curr_v=value[i:i+1]
                
                if i>half_bs:
                    resolution=int(hidden_states.shape[1]**0.5) 
                    
                    subject_k=key[half_bs:half_bs+1,identity_top_alpha_mask]
                    subject_v=value[half_bs:half_bs+1,identity_top_alpha_mask]
                    curr_k=torch.cat((curr_k,subject_k),dim=1)
                    curr_v=torch.cat((curr_v,subject_v),dim=1)
                
                curr_k = attn.head_to_batch_dim(curr_k).contiguous()
                curr_v = attn.head_to_batch_dim(curr_v).contiguous()
                
                scores = torch.matmul(curr_q, curr_k.transpose(-2, -1)) / (curr_k.shape[-1] ** 0.5)
                
                attention_weights = torch.nn.functional.softmax(scores, dim=-1)
                hidden_states = torch.matmul(attention_weights, curr_v)
                ex_out[start_idx:end_idx] = hidden_states
                    
            hidden_states = ex_out
            del ex_out
            
        else:  
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()
            
            # # attn_masks needs to be of shape [batch_size, query_tokens, key_tokens]
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, op=self.attention_op, scale=attn.scale
            )
            del query, key

        hidden_states = hidden_states.to(ori_dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if  not vanilla and curr_iter<transition_point:
            resolution=int(hidden_states.shape[1]**0.5)
            bs=subject_masks[resolution].shape[0]
            for i in range(1,bs):
                hidden_states[bs+i:bs+i+1,subject_masks[resolution][i]] = torch.einsum("cm,bmd->bcd", OT_plan[resolution][i-1], hidden_states[bs:bs+1,subject_masks[resolution][0]])
                
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        if optimalTransport is not None and curr_iter ==49:
            if self.curr_unet_part=="up" :
                image_line=hidden_states.shape[0]//2
                resolution=int(hidden_states.shape[1]**0.5)
                optimalTransport.set_fea(resolution,hidden_states[image_line:])
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def register_extended_self_attn(unet, attnstore):
    DICT_PLACE_TO_RES = {'down_0': 64, 'down_1': 64, 'down_2': 64, 'down_3': 64, 'down_4': 64, 'down_5': 64, 'down_6': 64, 'down_7': 64,
                         'down_8': 32, 'down_9': 32, 'down_10': 32, 'down_11': 32, 'down_12': 32, 'down_13': 32, 'down_14': 32, 'down_15': 32,
                         'down_16': 32, 'down_17': 32, 'down_18': 32, 'down_19': 32, 'down_20': 32, 'down_21': 32, 'down_22': 32, 'down_23': 32,
                         'down_24': 32, 'down_25': 32, 'down_26': 32, 'down_27': 32, 'down_28': 32, 'down_29': 32, 'down_30': 32, 'down_31': 32,
                         'down_32': 32, 'down_33': 32, 'down_34': 32, 'down_35': 32, 'down_36': 32, 'down_37': 32, 'down_38': 32, 'down_39': 32,
                         'down_40': 32, 'down_41': 32, 'down_42': 32, 'down_43': 32, 'down_44': 32, 'down_45': 32, 'down_46': 32, 'down_47': 32,
                         'mid_120': 32, 'mid_121': 32, 'mid_122': 32, 'mid_123': 32, 'mid_124': 32, 'mid_125': 32, 'mid_126': 32, 'mid_127': 32,
                         'mid_128': 32, 'mid_129': 32, 'mid_130': 32, 'mid_131': 32, 'mid_132': 32, 'mid_133': 32, 'mid_134': 32, 'mid_135': 32,
                         'mid_136': 32, 'mid_137': 32, 'mid_138': 32, 'mid_139': 32, 'up_49': 32, 'up_51': 32, 'up_53': 32, 'up_55': 32, 'up_57': 32,
                         'up_59': 32, 'up_61': 32, 'up_63': 32, 'up_65': 32, 'up_67': 32, 'up_69': 32, 'up_71': 32, 'up_73': 32, 'up_75': 32,
                         'up_77': 32, 'up_79': 32, 'up_81': 32, 'up_83': 32, 'up_85': 32, 'up_87': 32, 'up_89': 32, 'up_91': 32, 'up_93': 32,
                         'up_95': 32, 'up_97': 32, 'up_99': 32, 'up_101': 32, 'up_103': 32, 'up_105': 32, 'up_107': 32, 'up_109': 64, 'up_111': 64,
                         'up_113': 64, 'up_115': 64, 'up_117': 64, 'up_119': 64}
    attn_procs = {}
    
    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attn = (i % 2 == 0)

        if name.startswith("mid_block"):
            place_in_unet = f"mid_{i}"
        elif name.startswith("up_blocks"):
            place_in_unet = f"up_{i}"
        elif name.startswith("down_blocks"):
            place_in_unet = f"down_{i}"
        else:
            continue

        if is_self_attn:   
            attn_procs[name] = CoDiExtendedAttnXFormersAttnProcessor(place_in_unet, attnstore) 
        else:
            attn_procs[name] = CoDiAttnStoreProcessor(attnstore, place_in_unet) 

    unet.set_attn_processor(attn_procs)
