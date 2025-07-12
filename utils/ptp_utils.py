# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
# 
# Copyright (c) 2023 AttendAndExcite
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling ConsiStory or otherwise documented as NVIDIA-proprietary
# are not a contribution and subject to the license under the LICENSE file located at the root directory.

import torch
from collections import defaultdict
import numpy as np

from utils.general_utils import attn_map_to_binary
import torch.nn.functional as F

class AttentionStore:
    def __init__(self, attention_store_kwargs):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.attn_res = attention_store_kwargs.get('attn_res', (32,32))
        self.token_indices = attention_store_kwargs['token_indices']
        
        self.curr_iter = 0
        self.ALL_RES = [32, 64]
        self.step_store = defaultdict(list)
        self.attn_masks = {res: None for res in self.ALL_RES}
        self.last_mask = {res: None for res in self.ALL_RES}
        self.agg_attn_maps_store={}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, attn_heads: int):
        if is_cross and attn.shape[1] == np.prod(self.attn_res):
            
            guidance_attention = attn[attn.size(0)//2:]
            batched_guidance_attention = guidance_attention.reshape([guidance_attention.shape[0]//attn_heads, attn_heads, *guidance_attention.shape[1:]])
            batched_guidance_attention = batched_guidance_attention.mean(dim=1)
            self.step_store[place_in_unet].append(batched_guidance_attention)

    def reset(self):
        self.step_store = defaultdict(list)
        self.attn_masks = {res: None for res in self.ALL_RES}
        self.last_mask = {res: None for res in self.ALL_RES}

        torch.cuda.empty_cache()

    def aggregate_last_steps_attention(self) -> torch.Tensor:
        
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        
        attention_maps = torch.cat([torch.stack(x[-20:]) for x in self.step_store.values()]).mean(dim=0)
        self.step_store=defaultdict(list) 
        bsz, wh, _ = attention_maps.shape

        # Create attention maps for each concept token, for each batch item
        agg_attn_maps = []
        for i in range(bsz):
            curr_prompt_indices = []

            for concept_token_indices in self.token_indices: 
                
                if concept_token_indices[i] != -1:
                    curr_prompt_indices.append(attention_maps[i, :, concept_token_indices[i]].view(*self.attn_res))
            
            agg_attn_maps.append(torch.stack(curr_prompt_indices)) 
        
        # Downsample the attention maps to the target resolution
        # and create the attention masks, unifying masks across the different concepts
        for tgt_size in self.ALL_RES:
            tgt_agg_attn_maps = [F.interpolate(x.unsqueeze(1), size=tgt_size, mode='bilinear').squeeze(1) for x in agg_attn_maps] 
            
            to_store=[x.detach().cpu() for x in tgt_agg_attn_maps]
            self.agg_attn_maps_store[self.curr_iter]=to_store
            self.to_store_attn_map=tgt_agg_attn_maps
            
            tgt_agg_attn_maps=to_store
            attn_masks = []
            
            for batch_item_map in tgt_agg_attn_maps:
                
                if batch_item_map.requires_grad:
                    
                    batch_item_map=batch_item_map.clone().detach()
                concept_attn_masks = []

                for concept_maps in batch_item_map: 
                    concept_attn_masks.append(torch.from_numpy(attn_map_to_binary(concept_maps, 1.)).to(attention_maps.device).bool().view(-1))

                concept_attn_masks = torch.stack(concept_attn_masks, dim=0).max(dim=0).values 
                attn_masks.append(concept_attn_masks)
            
            attn_masks = torch.stack(attn_masks)
            self.last_mask[tgt_size] = attn_masks.clone() 
            