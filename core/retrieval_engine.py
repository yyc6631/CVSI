### Retrieval engine core logic

import json
import os
from typing import Optional, Tuple, List, Dict, Union
import argparse
import clip
import numpy as np
import utils.openai_client as openai_api
import pickle
import torch
import tqdm
import data_process.data_utils as data_utils
import utils.prompt_templates as prompts
from utils.phi_network import build_text_encoder, Phi
from transformers import CLIPTextModelWithProjection
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection
from utils.encode_with_pseudo_tokens import encode_with_pseudo_tokens_HF
if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32


class RetrievalEngine:
    """Retrieval engine core logic"""

    def __init__(self):
        # Delayed import to avoid circular imports
        from core.feature_extractor import FeatureExtractor
        self.feature_extractor = FeatureExtractor()
    
    @torch.no_grad()
    def generate_query_features(self, device: torch.device, args: argparse.Namespace, clip_model: clip.model.CLIP, blip_model: callable, query_dataset: torch.utils.data.Dataset, precomputed_dict: Dict[str, Union[str,None]], **kwargs) -> Tuple[torch.Tensor, List[str], list]:
        
        torch.cuda.empty_cache()
        batch_size = 32
        # Get preprocess from kwargs
        preprocess = kwargs.get('preprocess', None)

        # Use BLIP to generate reference image captions
        if precomputed_dict['captions'] is None or not os.path.exists(precomputed_dict['captions']):
            all_captions, relative_captions = [], []
            gt_img_ids, query_ids = [], []
            target_names, reference_names = [], []
            
            query_loader = torch.utils.data.DataLoader(
                dataset=query_dataset, batch_size=batch_size, num_workers=8, 
                pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)         
            query_iterator = tqdm.tqdm(query_loader, position=0, desc='Generating reference image captions...')
            
            for batch in query_iterator:
                blip_image = batch['blip_ref_img'].to(device) 
                reference_names.extend(batch['reference_name'])
                if 'fashioniq' not in args.dataset:
                    relative_captions.extend(batch['relative_caption'])
                else:
                    rel_caps = batch['relative_captions'] 
                    rel_caps = np.array(rel_caps).T.flatten().tolist()
                    relative_captions.extend([
                        f"{rel_caps[i].strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}" for i in range(0, len(rel_caps), 2)
                        ])
                                
                if 'target_name' in batch:
                    target_names.extend(batch['target_name']) 
                if 'gt_img_ids' in batch:
                    gt_img_ids.extend(batch['gt_img_ids'])
                if 'query_id' in batch:
                    query_ids.extend(batch['query_id'])
                
                query_iterator.set_postfix_str(f'Shape: {blip_image.size()}')
                with torch.no_grad():
                    for i in tqdm.trange(blip_image.size(0), position=1, desc='Iterating over batch', leave=False):
                        img = blip_image[i].unsqueeze(0) 
                        caption = blip_model.generate({'image': img},use_nucleus_sampling=True,num_captions=15,num_beams=15)
                        all_captions.append(caption) #[[],[],...]
                        
                
            if precomputed_dict['captions'] is not None:
                res_dict = {
                    'all_captions': all_captions,
                    'gt_img_ids': gt_img_ids,
                    'relative_captions': relative_captions,
                    'target_names': target_names,
                    'reference_names': reference_names,
                    'query_ids': query_ids
                }
                pickle.dump(res_dict, open(precomputed_dict['captions'], 'wb'))
        else:
            print(f'Loading precomputed image captions from {precomputed_dict["captions"]}!')
            res_dict = pickle.load(open(precomputed_dict['captions'], 'rb'))
            all_captions, gt_img_ids, relative_captions, target_names, reference_names, query_ids = res_dict.values()
        

        # Use LLM to modify captions, get modified_captions and added_content
        if precomputed_dict['mods'] is None or not os.path.exists(precomputed_dict['mods']):
            modified_captions = []
            added_content = []

            base_prompt = eval(args.llm_prompt)
            changed_prompt = eval(args.llm_changed_prompt)  # Extract prompt for added content

            for i in tqdm.trange(len(all_captions), position=0, desc='Generating modified captions...'):
                current_image_captions = all_captions[i]
                current_relative_caption = relative_captions[i]

                # Build complete prompt
                full_prompt = base_prompt + f"\nImage Content: {', '.join(current_image_captions)}\nInstruction: {current_relative_caption}\nEdited Description:"

                try:
                    modified_caption = openai_api.openai_completion(full_prompt, engine=args.openai_engine, api_key=args.openai_key)
                    modified_captions.append([modified_caption.strip()])
                except Exception as e:
                    print(f"Error generating modified caption: {e}")
                    modified_captions.append(current_image_captions)

                # Build prompt for extracting added objects
                changed_full_prompt = changed_prompt + f"\nImage Content: {', '.join(current_image_captions)}\nInstruction: {current_relative_caption}\nObjects most likely added:"

                try:
                    added_object = openai_api.openai_completion(changed_full_prompt, engine=args.openai_engine, api_key=args.openai_key)
                    added_content.append(added_object.strip())
                except Exception as e:
                    print(f"Error generating added content: {e}")
                    added_content.append("")
                                
            if precomputed_dict['mods'] is not None:
                dump_dict = {'base_caption':all_captions, 'instruction':relative_captions, 'modified_captions': modified_captions, 'added_content':added_content}
                json.dump(dump_dict, open(precomputed_dict['mods'], 'w'), indent=6)

        else: 
            print(f'Loading precomputed caption modifiers from {precomputed_dict["mods"]}!')
            modified_captions = json.load(open(precomputed_dict['mods'], 'r'))['modified_captions']
            added_content = json.load(open(precomputed_dict['mods'], 'r'))['added_content']


        ### Generate pseudo tokens by mapping images
        # Parameters for phi mapping
        mixed_precision = 'fp16'

        # Automatically select corresponding phi model file based on args.clip parameter
        if args.clip == 'ViT-B/32':
            phi_checkpoint_name = './phi_model/phi_best_base_openai.pt'
            clip_model_name = "base_openai"
        elif args.clip == 'ViT-L/14':
            phi_checkpoint_name = './phi_model/phi_best_large_openai.pt'
            clip_model_name = "large_openai"
        elif args.clip == 'ViT-bigG-14':
            phi_checkpoint_name = './phi_model/phi_best_giga_openclip.pt'
            clip_model_name = "giga"
        else:
            # Default to large model as fallback
            phi_checkpoint_name = './phi_model/phi_best_large_openai.pt'
            clip_model_name = "large_openai"

        cache_dir = "./hf_models"

        # Fully load pre-trained phi model
        image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(clip_model_name,mixed_precision,cache_dir)

        phi = Phi(input_dim=text_encoder.config.projection_dim,
                  hidden_dim=text_encoder.config.projection_dim * 4,
                  output_dim=text_encoder.config.hidden_size, dropout=0.5).to(device)
        phi.load_state_dict(torch.load(phi_checkpoint_name, map_location=device)[phi.__class__.__name__])
        phi = phi.eval()
        image_encoder = image_encoder.float().to(device)
        text_encoder = text_encoder.float().to(device)

        # Get global image pseudo tokens
        query_loader = torch.utils.data.DataLoader(
                dataset=query_dataset, batch_size=batch_size, num_workers=8,
                pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)
        pseudo_tokens,ref_names_list = self.feature_extractor.generate_pseudo_tokens(image_encoder,phi,query_loader,device)
        pseudo_tokens = pseudo_tokens.to(device)
               
        new_text_relative = []
        
        for i in range(len(modified_captions)):
            new_text_relative.append("a photo of $ that "+relative_captions[i]+". And the photo should have "+added_content[i]+".")

        # Get Average modified captions features
        text_features = self.feature_extractor.encode_text_with_averaging(device, clip_model, modified_captions, batch_size=batch_size)

        reference_features = []
        query_loader = torch.utils.data.DataLoader(
            dataset=query_dataset, batch_size=batch_size, num_workers=8,
            pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)
        query_iterator = tqdm.tqdm(query_loader, position=0, desc='Generating reference_image features...')
        for batch in query_iterator:
            images = batch.get('image')
            names = batch.get('image_name')
            if images is None: images = batch.get('reference_image')
            if names is None: names = batch.get('reference_name')
            images = images.to(device)
            query_iterator.set_postfix_str(f'Shape: {images.size()}')
            with torch.no_grad():
                global_feat = clip_model.encode_image(images)
                global_feat = F.normalize(global_feat,p=2, dim=-1)
                reference_features.append(global_feat.cpu())

        reference_features = torch.vstack(reference_features)



        ### Get Fine-grained features
        tokenized_in_captions = clip.tokenize(new_text_relative, context_length=77,truncate=True).to(device)
        # Split tokenized_in_captions into multiple small batches (to avoid memory overflow)
        num_batches = (tokenized_in_captions.shape[0] + batch_size - 1) // batch_size
        all_text_features_hf = []
        # Process each batch
        for i in range(num_batches):
            # Get current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, tokenized_in_captions.shape[0])
            tokenized_batch = tokenized_in_captions[start_idx:end_idx]
            # Get corresponding pseudo_tokens (assuming pseudo_tokens size is [batch_size, n_ctx])
            pseudo_tokens_batch = pseudo_tokens[start_idx:end_idx]  # Select corresponding pseudo_tokens based on batch size
            # Call encode_with_pseudo_tokens_HF function to process current batch
            text_features_batch = encode_with_pseudo_tokens_HF(text_encoder, tokenized_batch, pseudo_tokens_batch)
            text_features_batch = F.normalize(text_features_batch)
            # Add current batch features to all batch features list
            all_text_features_hf.append(text_features_batch)
        # Merge all batch features
        text_features_hf = torch.cat(all_text_features_hf, dim=0)
        # Move result to correct device
        text_features_hf = text_features_hf.to(device)



        return {
            'predicted_features_1': text_features,  # Average modified captions features
            'predicted_features_2': text_features_hf, # fine-grained features
            'predicted_features_3': reference_features, # reference image features
            'target_names': target_names,
            'targets': gt_img_ids,
            'reference_names': reference_names,
            'query_ids': query_ids,
            'start_captions': all_captions,
            'modified_captions': modified_captions,
            'instructions': relative_captions
        }
