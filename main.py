import argparse
import os
from codi_utils import load_pipeline, create_latents, create_token_indices, OptimalTransport
from utils.general_utils import *
import torch
from PIL import Image
import torch.nn.functional as F
import time
LATENT_RESOLUTIONS = [32, 64]


def CoDi_generation(args,story_pipeline, prompts, concept_token,
                        seed,n_steps=50):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    
    batch_size = len(prompts)
    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)
    attn_res=(32,32)
    default_attention_store_kwargs = {
        'token_indices': token_indices,
        'attn_res':attn_res
    }
    latents, g = create_latents(story_pipeline, seed, batch_size, args.same_latent, device, float_type) 
    optimalTransport = OptimalTransport()
    images=story_pipeline(prompt=prompts, generator=g, latents=latents, 
                        attention_store_kwargs=default_attention_store_kwargs,
                        record_attention=True,
                        vanilla=True,
                        num_inference_steps=n_steps,
                        optimalTransport=optimalTransport).images
    
    subject_masks = story_pipeline.attention_store.last_mask 

    optimalTransport.set_subject_mask(subject_masks)
    
    sim_matrix=None
    attn_map={}
    
    attn_map_32=[F.interpolate(x.mean(dim=0,keepdim=True).unsqueeze(1), size=32, mode='bilinear').squeeze(1).squeeze(0).reshape(-1) for x in story_pipeline.attention_store.to_store_attn_map]
    attn_map[32]=attn_map_32
    attn_map_64=[x.mean(dim=0,keepdim=True).squeeze(0).reshape(-1) for x in story_pipeline.attention_store.to_store_attn_map]
    attn_map[64]=attn_map_64
    optimalTransport.set_attn_map(attn_map)
    
    OT_plan,sim_matrix=optimalTransport.get_OT_plan()
    identity_top_alpha_masks={}
    
    for resolution in sim_matrix.keys():
        M_id=subject_masks[resolution][0]
        thresholded = [torch.mul(f, s) for f, s in zip(OT_plan[resolution], sim_matrix[resolution])]
        
        summed = [t.sum(dim=0, keepdim=True) for t in thresholded]
        stacked = torch.cat(summed, dim=0)
        
        token_influence = torch.sum(stacked, dim=0)
        k = int(M_id.sum() * args.alpha) 
        _, topk_indices = torch.topk(token_influence, k=k)

        true_indices = torch.nonzero(M_id).squeeze()  
        topk_absolute_indices = true_indices[topk_indices]  
        identity_mask = torch.zeros_like(M_id)
        identity_mask[topk_absolute_indices] = True
        identity_top_alpha_masks[resolution]=identity_mask
    
    out = story_pipeline(prompt=prompts, generator=g, latents=latents,
                        attention_store_kwargs=default_attention_store_kwargs,
                        record_attention=False,
                        num_inference_steps=n_steps,
                        args=args,
                        subject_masks=subject_masks,
                        OT_plan=OT_plan,
                        identity_top_alpha_masks=identity_top_alpha_masks,
                        transition_point=args.transition_point,
                        )
    images=out.images
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int, required=False)
    parser.add_argument('--seed', default=40, type=int, required=False)
    parser.add_argument('--same_latent', default=False, type=bool, required=False, help="different latent to ensure different pose")
    parser.add_argument('--style', default="A 3D animation of", type=str, required=False)
    parser.add_argument('--subject', default="A happy hedgehog", type=str, required=False)
    parser.add_argument('--concept_token', default=["hedgehog"],
                        type=str, nargs='*', required=False)
    parser.add_argument('--settings', default=["in a cozy nest","dressed in a miniature jacket",
                                               "wearing a small collar", "dressed in a festive outfit",
                                               "wearing a flower crown"], 
                        type=str, nargs='*', required=False)
    parser.add_argument('--transition_point', type=int, default=10, required=False)
    parser.add_argument('--alpha', type=float, default=0.5, required=False)
    parser.add_argument('--root_dir', default="./result",type=str, required=False)
    args = parser.parse_args()
    
    story_pipeline = load_pipeline(args.gpu)
    identity_prompt=f'{args.style} {args.subject}'

    save_dir=os.path.join(args.root_dir,str(time.time()))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    prompts=[]
    prompts.append(identity_prompt)
    
    for setting in args.settings:
        prompts.append(f'{args.style} {args.subject} {setting}')
    images=CoDi_generation(args,story_pipeline,prompts,args.concept_token,args.seed)
    
    story_images=[]
    visual_prompts="Identity Prompt:"
    for i in range(len(images)):
        visual_prompts+=f"{prompts[i]}" if i ==0 else f" t{i}:{prompts[i]}".replace(identity_prompt,"")
        if i!=0:
            images[i].save(f"{os.path.join(save_dir,prompts[i])}.jpg")
            story_images.append(np.array(images[i]))
    concatenated_image = np.concatenate(story_images, axis=1)
    concatenated_image1=text_under_image(concatenated_image,visual_prompts,font_scale=2.1,add_h=0.1)
    img=Image.fromarray(concatenated_image1)
    img.save(f"{os.path.join(save_dir,'story')}.jpg")

