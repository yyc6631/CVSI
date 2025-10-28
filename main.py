import os
from typing import List, Dict
import argparse
import clip
import lavis
import numpy as np
import termcolor
import torch
import compute_results
import data_process.data_utils as data_utils
import data_process.datasets as datasets
import utils.prompt_templates as prompts
from core.feature_extractor import FeatureExtractor
from core.retrieval_engine import RetrievalEngine
from config.experiment_config import ExperimentConfig
import pickle
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--clip", type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-L/14','ViT-bigG-14'])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['cirr', 'circo','fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt'])
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'])
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--weight-path", type=str, default='')
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'])
    available_prompts = [f'prompts.{x}' for x in prompts.__dict__.keys() if '__' not in x]
    parser.add_argument("--llm_prompt", default='prompts.contextual_modifier_prompt', type=str, choices=available_prompts)
    parser.add_argument("--llm_changed_prompt", default='prompts.re_and_add_prompt', type=str)
    parser.add_argument("--openai_engine", default='gpt-3.5-turbo', type=str, choices=['gpt-3.5-turbo', 'gpt-4o'])
    parser.add_argument("--openai_key", default="<your openai_key>", type=str)
    args = parser.parse_args()

    # Set fixed parameter values
    args.precomputed = ['img_features', 'captions', 'mods', 'target_captions']
    args.blip = 'blip2_opt'

    ### Set Device.
    termcolor.cprint(f'Starting evaluation on {args.dataset.upper()} (split: {args.split})\n', color='green', attrs=['bold'])
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    ### Initialize configuration manager
    config = ExperimentConfig(args)
    precomputed_dict = config.get_precomputed_paths()


    # Load CLIP model and preprocess
    print(f'Loading CLIP {args.clip}... ', end='')
          
    if args.clip in ['ViT-bigG-14']:
        import open_clip
        pretraining = {
            'ViT-bigG-14':'laion2b_s39b_b160k'
        }
        if args.weight_path == '':
            weight_path = os.path.join(args.dataset_path, '..', 'weights', 'open_clip')
        else:
            weight_path = os.path.join(os.getcwd(), args.weight_path)
        os.makedirs(weight_path, exist_ok=True)
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip, pretrained=pretraining[args.clip], cache_dir=weight_path)
        clip_model = clip_model.eval().requires_grad_(False).to(device)
        tokenizer = open_clip.get_tokenizer(args.clip)
        clip_model.tokenizer = tokenizer
    else:
        clip_model, clip_preprocess = clip.load(args.clip, device=device, jit=False)
        clip_model = clip_model.float().eval().requires_grad_(False).to(device)

    print('Done.')


    # Preprocess, convert to CLIP standard input
    if args.preprocess_type == 'targetpad':
        print('Target pad preprocess pipeline is used.')
        preprocess = data_utils.targetpad_transform(1.25, clip_preprocess.transforms[0].size)
    elif args.preprocess_type == 'clip':
        print('CLIP preprocess pipeline is used.')
        preprocess = clip_preprocess



    # Load BLIP model
    blip_model = None
    if precomputed_dict['captions'] is None or not os.path.exists(precomputed_dict['captions']) or precomputed_dict['target_captions'] is None or not os.path.exists(precomputed_dict['target_captions']) :
        blip_model, vis_processors, _ = lavis.models.load_model_and_preprocess(
            name=args.blip, model_type="caption_coco_opt6.7b", is_eval=True, device=device)
    else:
        import omegaconf
        model_cls = lavis.common.registry.registry.get_model_class(args.blip)
        preprocess_cfg = omegaconf.OmegaConf.load(model_cls.default_config_path("caption_coco_opt6.7b")).preprocess
        vis_processors, _ = lavis.models.load_preprocess(preprocess_cfg)
        print(f'Skipped loading of BLIP ({args.blip}).')

    ### Initialize core components
    feature_extractor = FeatureExtractor()
    retrieval_engine = RetrievalEngine()

    # Load datasets (database, query datasets, types)
    target_datasets, query_datasets, pairings = [], [], []

    if 'fashioniq' in args.dataset.lower():
        dress_type = args.dataset.split('_')[-1]
        target_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'classic', preprocess, blip_transform=vis_processors["eval"]))
        query_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'relative', preprocess, blip_transform=vis_processors["eval"]))
        pairings.append(dress_type)
        compute_results_function = compute_results.fiq

    elif args.dataset.lower() == 'cirr':
        split = 'test1' if args.split == 'test' else args.split
        target_datasets.append(datasets.CIRRDataset(args.dataset_path, split, 'classic', preprocess, blip_transform=vis_processors['eval']))
        query_datasets.append(datasets.CIRRDataset(args.dataset_path, split, 'relative', preprocess, blip_transform=vis_processors["eval"]))
        compute_results_function = compute_results.cirr
        pairings.append('default')

    elif args.dataset.lower() == 'circo':
        target_datasets.append(datasets.CIRCODataset(args.dataset_path, args.split, 'classic', preprocess, blip_transform=vis_processors["eval"]))
        query_datasets.append(datasets.CIRCODataset(args.dataset_path, args.split, 'relative', preprocess, blip_transform=vis_processors["eval"]))
        compute_results_function = compute_results.circo
        pairings.append('default')
    

    # Evaluate performance, compute metrics
    for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
        termcolor.cprint(f'\n------ Evaluating Retrieval Setup: {pairing}', color='yellow', attrs=['bold'])


        input_kwargs = {
            'args': args, 'query_dataset': query_dataset, 'target_dataset': target_dataset, 'clip_model': clip_model,
            'blip_model': blip_model, 'preprocess': preprocess, 'device': device, 'split': args.split,
            'blip_transform': vis_processors['eval'], 'precomputed_dict': precomputed_dict,
        }

        ### Compute Target Image Features
        print(f'Extracting target image features using CLIP: {args.clip}.')
        index_features, index_names = feature_extractor.extract_clip_image_features(
            device, args, target_dataset, clip_model, precomputed=precomputed_dict['img_features'])
        index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)
        input_kwargs.update({'index_features': index_features, 'index_names': index_names})

        ### Compute Target Image Caption Features
        print(f'generating target image caption ')
        feature_extractor.generate_blip_captions(device, args, target_dataset, clip_model,blip_model,precomputed=precomputed_dict['target_captions'])


        # Encode the obtained target_captions to get corresponding features
        with open(precomputed_dict['target_captions'], 'rb') as f:
            data = pickle.load(f)
        target_captions = data['target_captions']
        index_names_2 = data['index_names']
        if args.clip =='ViT-B/32':
            backbone = 'base'
        elif args.clip =='ViT-L/14':
            backbone = 'large'
        elif args.clip =='ViT-bigG-14':
            backbone = 'giga'
        else :
            backbone = 'None'

        # Read target_captions_features
        target_captions_features_path = config.get_target_captions_features_path()
        if os.path.exists(target_captions_features_path):
            with open(target_captions_features_path, 'rb') as f:
                target_captions_features = pickle.load(f)
        else:
            target_captions_features = feature_extractor.encode_text_with_averaging(device,clip_model,target_captions)
            with open(target_captions_features_path, 'wb') as f:
                pickle.dump(target_captions_features, f)
        input_kwargs.update({'target_captions_features': target_captions_features})

        ### Compute Method-specific Query Features.
        print(f'Generating conditional query predictions (CLIP: {args.clip}, BLIP: {args.blip}).')

        # Update parameters
        out_dict = retrieval_engine.generate_query_features(**input_kwargs)
        input_kwargs.update(out_dict)

        # Compute retrieval metrics
        print('Computing final retrieval metrics.')
        result_metrics = compute_results_function(**input_kwargs)
        
        # Print metrics.
        print('\n')
        if result_metrics is not None:
            termcolor.cprint(f'Metrics for {args.dataset.upper()} ({args.split})- {pairing}', attrs=['bold'])
            for k, v in result_metrics.items():
                print(f"{pairing}_{k} = {v:.2f}")        
        else:
            termcolor.cprint(f'No explicit metrics available for {args.dataset.upper()} ({args.split}) - {pairing}.', attrs=['bold'])            



if __name__ == '__main__':
    main()
