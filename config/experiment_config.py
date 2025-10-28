### Experiment configuration management

import os
from typing import Dict, Union


class ExperimentConfig:
    """Experiment configuration management"""

    def __init__(self, args):
        self.args = args
        self.precomputed_dict = {key: None for key in ['img_features', 'captions', 'mods']}
        self._setup_precomputed_paths()

    def _setup_precomputed_paths(self):
        """Set up precomputed file paths (paths stored in precomputed_dict)"""
        if self.args.clip in ['ViT-L/14','ViT-B/32']:
            precomputed_str = f'{self.args.dataset}_{self.args.clip}_openai_{self.args.split}'.replace('/', '-')
        else:
            precomputed_str = f'{self.args.dataset}_{self.args.clip}_{self.args.split}'.replace('/', '-')

        if len(self.args.precomputed):
            os.makedirs('precomputed', exist_ok=True)

        if 'img_features' in self.args.precomputed:
            self.precomputed_dict['img_features'] = os.path.join('precomputed', precomputed_str + '_img_features.pkl')
        if 'captions' in self.args.precomputed:
            caption_load_str = f'{self.args.dataset}_{self.args.split}'.replace('/', '-')
            self.precomputed_dict['captions'] = os.path.join('precomputed', caption_load_str + '_captions.pkl')

        if 'target_captions' in self.args.precomputed:
            caption_load_str = f'{self.args.dataset}_{self.args.split}'.replace('/', '-')
            self.precomputed_dict['target_captions'] = os.path.join('precomputed', caption_load_str + '_target_captions.pkl')

        if 'mods' in self.args.precomputed:
            mod_load_str = f'{self.args.dataset}_{self.args.split}'.replace('/', '-')
            self.precomputed_dict['mods'] = os.path.join('precomputed', mod_load_str + f'_mods_{self.args.llm_prompt.split(".")[-1]}.json')
            if self.args.openai_engine != 'gpt-3.5-turbo':
                self.precomputed_dict['mods'] = self.precomputed_dict['mods'].replace('.json', f'_{self.args.openai_engine}.json')

        if self.args.split == 'test':
            self.precomputed_dict['test'] = precomputed_str + f'{self.args.llm_prompt.split(".")[-1]}_test_submission.json'
    
    def get_precomputed_paths(self) -> Dict[str, Union[str, None]]:
        """Get precomputed file paths"""
        return self.precomputed_dict

    def get_target_captions_features_path(self) -> str:
        """Get target caption features file path"""
        if self.args.clip =='ViT-B/32':
            backbone = 'base'
        elif self.args.clip =='ViT-L/14':
            backbone = 'large'
        elif self.args.clip =='ViT-bigG-14':
            backbone = 'giga'
        else :
            backbone = 'None'
        
        os.makedirs('./target_captions_features', exist_ok=True)
        return f'./target_captions_features/{self.args.dataset}_{backbone}_target_captions_features.pkl'
