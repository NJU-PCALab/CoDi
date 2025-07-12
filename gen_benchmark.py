import argparse
import os
import yaml
from codi_utils import load_pipeline, create_latents, create_token_indices, OptimalTransport
from utils.general_utils import *
import torch
import queue 
import threading
import gc
from PIL import Image
from typing import Any
import torch.nn.functional as F
LATENT_RESOLUTIONS = [32, 64]


def CoDi_generation(args,story_pipeline, prompts, concept_token,
                        seed, subject_domain,n_steps=50):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    
    batch_size = len(prompts)
    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)
    if "animals" in subject_domain:
        attn_res=(64,64)
    else:
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
        thresholded = [tensor * (tensor > 0).float() for tensor in OT_plan[resolution]]
        
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

def run_batch(args,gpu, seed,
              style, subject, concept_token,
              settings,
              save_dir = None):
    
    story_pipeline = load_pipeline(gpu)
    identity_prompt=f'{style} {subject}'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    prompts=[]
    prompts.append(identity_prompt)
    
    subject_domain=save_dir.split("/")[-1]
    for setting in settings:
        prompts.append(f'{style} {subject} {setting}')
    images=CoDi_generation(args,story_pipeline,prompts,concept_token,seed,subject_domain)
    
    story_images=[]
    visual_prompts=visual_prompts="Identity Prompt:"
    for i in range(len(images)):
        visual_prompts+=f"{prompts[i]}" if i ==0 else f" t{i}:{prompts[i]}".replace(identity_prompt,"")
        if i!=0:
            images[i].save(f"{os.path.join(save_dir,prompts[i])}.jpg")
            story_images.append(np.array(images[i]))
    concatenated_image = np.concatenate(story_images, axis=1)
    concatenated_image1=text_under_image(concatenated_image,visual_prompts,font_scale=2.1,add_h=0.1)
    img=Image.fromarray(concatenated_image1)
    img.save(f"{os.path.join(save_dir,'story')}.jpg")
    print(f"{os.path.join(save_dir,'story')}.jpg")
    
    return None
    
def worker(device, task_queue,args):
    # Process tasks until queue is empty
    while not task_queue.empty():
        instance = task_queue.get()
        if instance is None:  # If None is encountered, stop the worker
            break
        
        result = run_batch(args,device, instance[-1], instance[1], 
                  instance[2], instance[4], instance[3], instance[0])
        
        del result  
        torch.cuda.empty_cache()
        gc.collect()  
        print(f"Finished processing {instance[1]}")
        task_queue.task_done() 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=int, default=[0], nargs='+', help='List of device indices')
    parser.add_argument('--seed', default=40, type=int, required=False)
    parser.add_argument('--same_latent', default=False, type=bool, required=False)
    parser.add_argument('--benchmark_path', type=str,default="./resource/consistory+.yaml")
    parser.add_argument('--transition_point', type=int, default=10, required=False)
    parser.add_argument('--alpha', type=float, default=0.5, required=False)
    parser.add_argument('--number',default=0,type=int, required=False)
    parser.add_argument('--root_dir', default="./result",type=str, required=False)
    args = parser.parse_args()
    devices=args.devices
    
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    
    save_dir=os.path.join(args.root_dir,"benchmark")
    dir_name=f"{args.number}_TransitionPoint{args.transition_point}_alpha{args.alpha}"
    save_dir=os.path.join(save_dir,dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.expanduser(args.benchmark_path), 'r') as file:
        data = yaml.safe_load(file)
    
    instances = []
    for subject_domain, subject_domain_instances in data.items():
        for index, instance in enumerate(subject_domain_instances):
            id_prompt = f'{instance["style"]} {instance["subject"]}'
            frame_prompt_list = instance["settings"]
            save_dir_subject_domain = os.path.join(save_dir, f"{subject_domain}_{index}")
            if args.seed != -1:
                seed = args.seed
            else:
                import random
                seed = random.randint(0, 2**32 - 1)
            instances.append((save_dir_subject_domain, instance["style"], instance["subject"],instance["settings"],instance["concept_token"], seed))
    
    # Create a task queue and populate it with instances
    task_queue = queue.Queue()
    for instance in instances:
        task_queue.put(instance)
    
    threads = []
    for device in devices:
        thread = threading.Thread(target=worker, args=(device, task_queue,args))
        threads.append(thread)
        thread.start()
        
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
