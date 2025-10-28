### Feature extraction related functions

import os
from typing import Optional, Tuple, List, Dict, Union
import argparse
import clip
import numpy as np
import pickle
import torch
import tqdm
import data_process.data_utils as data_utils
from utils.phi_network import build_text_encoder, Phi
from transformers import CLIPTextModelWithProjection
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection

if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32


class FeatureExtractor:
    """Unified feature extraction interface"""
    
    @torch.no_grad()
    def extract_clip_image_features(self, device: torch.device, args: argparse.Namespace, dataset: torch.utils.data.Dataset, clip_model: clip.model.CLIP, batch_size: Optional[int] = 32,
                               num_workers: Optional[int] = 8, precomputed: str=None, **kwargs) -> Tuple[torch.Tensor, List[str]]:
        """
        Extracts image features from a dataset using a CLIP model.
        """

        if precomputed is not None and os.path.exists(precomputed):
            print(f'Loading precomputed image features from {precomputed}!')
            extracted_data = pickle.load(open(precomputed, 'rb'))
            index_features, index_names = extracted_data['index_features'], extracted_data['index_names']
        else:
            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=True, collate_fn=data_utils.collate_fn)

            index_features, index_names = [], []
                
            try:
                print(f"Extracting image features {dataset.__class__.__name__} - {dataset.split}")
            except Exception as e:
                pass

            # Extract features    
            index_rank = None
            for batch in tqdm.tqdm(loader):
                images = batch.get('image')
                names = batch.get('image_name')
                if images is None: images = batch.get('reference_image')
                if names is None: names = batch.get('reference_name')

                images = images.to(device)
                with torch.no_grad(),torch.cuda.amp.autocast():
                    batch_features = clip_model.encode_image(images)
                    index_features.append(batch_features.cpu())
                    index_names.extend(names)
            
            index_features = torch.vstack(index_features)

            if precomputed is not None:
                pickle.dump({'index_features': index_features, 'index_names': index_names}, open(precomputed, 'wb'))
        
        return index_features, index_names

    # This generation specifically refers to target_captions generation
    @torch.no_grad()
    def generate_blip_captions(self, device: torch.device, args: argparse.Namespace, dataset: torch.utils.data.Dataset, clip_model: clip.model.CLIP, blip_model,batch_size: Optional[int] = 32,
                               num_workers: Optional[int] = 8, precomputed: str=None, **kwargs) -> Tuple[torch.Tensor, List[str]]:

        # Here precomputed specifically refers to precomputed_dict['target_captions']
        if precomputed is not None and os.path.exists(precomputed):
            print(f'get target captions from {precomputed}!')
            target_data = pickle.load(open(precomputed, 'rb'))
            target_captions, index_names = target_data['target_captions'], target_data['index_names']
        else:
            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=True, collate_fn=data_utils.collate_fn)
            target_captions = []
            XXXXX_features, index_names, = [], []

            try:
                print(f"get target captions {dataset.__class__.__name__} - {dataset.split}")
            except Exception as e:
                pass

            for batch in tqdm.tqdm(loader):
                # Get target images and corresponding names from target_dataset
                blip_image = batch.get('blip_img')
                names = batch.get('image_name')
                blip_image = blip_image.to(device)
                with torch.no_grad():
                    for i in tqdm.trange(blip_image.size(0), position=1, desc='Iterating over batch', leave=False):
                        index_names.extend(names)  # Store image names
                        img = blip_image[i].unsqueeze(0)
                        caption = blip_model.generate({'image': img},use_nucleus_sampling=True,num_captions=15,num_beams=15)
                        print("this target image's 15 captions are:",caption)
                        target_captions.append(caption)  # [[],[],...]
                    

            if precomputed is not None: 
                target_data = {
                    'target_captions': target_captions,
                    'index_names': index_names
                }
                pickle.dump(target_data, open(precomputed, 'wb'))

    @torch.no_grad()
    def generate_pseudo_tokens(self, clip_model: CLIPVisionModelWithProjection, phi: Phi, data_loader, device) -> Tuple[torch.Tensor, List[str]]:
        
        predicted_tokens = []
        names_list = []
        print(f"Extracting tokens using phi model")
        for batch in tqdm.tqdm(data_loader):
            images = batch.get('image')
            names = batch.get('image_name')
            if images is None:
                images = batch.get('reference_image')
            if names is None:
                names = batch.get('reference_name')

            images = images.to(device)
            image_features = clip_model(pixel_values=images.half()).image_embeds
            batch_predicted_tokens = phi(image_features) 
            predicted_tokens.append(batch_predicted_tokens.cpu())
            names_list.extend(names)

        predicted_tokens = torch.vstack(predicted_tokens)
        return predicted_tokens, names_list

    def encode_text_with_averaging(self, device, clip_model, input_captions, batch_size=32):
        """Text feature extraction function, can be used to extract all caption features"""
        n_iter = int(np.ceil(len(input_captions)/batch_size))
        modified_features = []

        for i in tqdm.trange(n_iter, position=0, desc='Encoding captions...'):
            captions_to_use = input_captions[i*batch_size:(i+1)*batch_size]
            for j in range(len(captions_to_use)):
                current_image_captions = captions_to_use[j]
                if hasattr(clip_model, 'tokenizer'):
                    tokenized_input_captions = clip_model.tokenizer(current_image_captions, context_length=77).to(device)
                else:
                    tokenized_input_captions = clip.tokenize(current_image_captions, context_length=77, truncate=True).to(device)
                clip_text_features = clip_model.encode_text(tokenized_input_captions)
                clip_text_features = F.normalize(clip_text_features, dim=-1)
                clip_text_features = torch.mean(clip_text_features, dim=0, keepdim=True)
                modified_features.append(clip_text_features.cpu())

        return torch.vstack(modified_features)
