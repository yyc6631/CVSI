import json
import os
from typing import List, Dict, Union

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import tqdm
import itertools
# Calculate FashionIQ corresponding metrics
@torch.no_grad()
def fiq(
    device: torch.device,
    predicted_features_1: torch.Tensor,  # Average modified captions features
    predicted_features_2: torch.Tensor,  #fine-grained features
    predicted_features_3: torch.Tensor,  #reference image features
    target_names: List,
    index_features: torch.Tensor,  #target image features
    target_captions_features: torch.Tensor, #target captions features
    index_names: List,
    split: str='val',
    **kwargs
) -> Dict[str, float]:
    
    index_features = torch.nn.functional.normalize(index_features).to(device)
    target_captions_features = torch.nn.functional.normalize(target_captions_features).to(device)
    predicted_features_1 = predicted_features_1.to(device)
    predicted_features_2 = predicted_features_2.to(device)
    predicted_features_3 = predicted_features_3.to(device)

    predicted_features = predicted_features_1*0.2+predicted_features_2*0.6+predicted_features_3*0.2
    distances = 1 - predicted_features @ (index_features.T*0.9+target_captions_features.T*0.1) 



    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    # Compute the metrics
    output_metrics = {
        'Recall@1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        'Recall@5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        'Recall@10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        'Recall@50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    }
    return output_metrics
    

@torch.no_grad()
def cirr(
    device: torch.device,
    predicted_features_1: torch.Tensor,
    predicted_features_2: torch.Tensor,
    predicted_features_3: torch.Tensor,
    reference_names: List,
    targets: Union[np.ndarray,List],
    target_names: List,
    index_features: torch.Tensor,
    target_captions_features: torch.Tensor,
    index_names: List,
    query_ids: Union[np.ndarray,List],
    precomputed_dict: Dict[str, Union[str, None]],
    split: str='val',
    **kwargs
) -> Dict[str, float]:

    # Put on device.
    index_features = index_features.to(device)
    target_captions_features = target_captions_features.to(device)
    predicted_features_1 = predicted_features_1.to(device)
    predicted_features_2 = predicted_features_2.to(device)
    predicted_features_3 = predicted_features_3.to(device)
    # Compute the distances and sort the results
    distances = 1 - (predicted_features_1*0.6+predicted_features_2*0.4) @ (index_features.T*0.8+target_captions_features.T*0.2)
    if distances.ndim == 3:
        # If there are multiple features per instance, we average.
        distances = distances.mean(dim=1)
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    resize = len(sorted_index_names) if split == 'test' else len(target_names)
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(resize, -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    
    # Compute the subset predictions and ground-truth labels
    targets = np.array(targets)
    group_mask = (sorted_index_names[..., None] == targets[:, None, :]).sum(-1).astype(bool)

    if split == 'test':
        sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)
        pairid_to_retrieved_images, pairid_to_group_retrieved_images = {}, {}
        for pair_id, prediction in zip(query_ids, sorted_index_names):
            pairid_to_retrieved_images[str(int(pair_id))] = prediction[:50].tolist()
        for pair_id, prediction in zip(query_ids, sorted_group_names):
            pairid_to_group_retrieved_images[str(int(pair_id))] = prediction[:3].tolist()            

        submission = {'version': 'rc2', 'metric': 'recall'}
        group_submission = {'version': 'rc2', 'metric': 'recall_subset'}

        submission.update(pairid_to_retrieved_images)
        group_submission.update(pairid_to_group_retrieved_images)

        submissions_folder_path = os.path.join(os.getcwd(), 'results', 'CVSI_results', 'cirr')
        os.makedirs(submissions_folder_path, exist_ok=True)

        with open(os.path.join(submissions_folder_path, precomputed_dict['test']), 'w') as file:
            json.dump(submission, file, sort_keys=True)
        with open(os.path.join(submissions_folder_path, f"subset_{precomputed_dict['test']}"), 'w') as file:
            json.dump(group_submission, file, sort_keys=True)                        
        return None
            
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))    
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    output_metrics = {f'recall@{key}': (torch.sum(labels[:, :key]) / len(labels)).item() * 100 for key in [1, 5, 10, 50]}
    output_metrics.update({f'group_recall@{key}': (torch.sum(group_labels[:, :key]) / len(group_labels)).item() * 100 for key in [1, 2, 3]})

    return output_metrics


@torch.no_grad()
def circo(
    device: torch.device,
    predicted_features_1: torch.Tensor,
    predicted_features_2: torch.Tensor,
    predicted_features_3: torch.Tensor,
    targets: Union[np.ndarray,List],
    target_names: List,
    index_features: torch.Tensor,
    target_captions_features: torch.Tensor,
    index_names: List,
    query_ids: Union[np.ndarray,List],
    precomputed_dict: Dict[str, Union[str, None]],
    split: str='val',
    **kwargs
) -> Dict[str, float]:
    
    # Put on device
    index_features = index_features.to(device)
    target_captions_features = target_captions_features.to(device)
    predicted_features_1 = predicted_features_1.to(device)
    predicted_features_2 = predicted_features_2.to(device)
    predicted_features_3 = predicted_features_3.to(device)
    ### Compute Test Submission in case of test split.
    if split == 'test':
        print('Generating test submission file!')
        similarity = (predicted_features_1*0.6+predicted_features_2*0.4) @ (index_features.T*0.8+target_captions_features.T*0.2) 
        
        if similarity.ndim == 3:
            # If there are multiple features per instance, we average.
            similarity = similarity.mean(dim=1)                    
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu() 
        sorted_index_names = np.array(index_names)[sorted_indices] 
        # Return prediction dict to submit.
        queryid_to_retrieved_images = {
            query_id: query_sorted_names[:50].tolist() for (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)            
        }
        
        submissions_folder_path = os.path.join(os.getcwd(), 'results', 'CVSI_results', 'circo')
        os.makedirs(submissions_folder_path, exist_ok=True)
        with open(os.path.join(submissions_folder_path, precomputed_dict['test']), 'w') as file:
            json.dump(queryid_to_retrieved_images, file, sort_keys=True)        
        return None
        

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
def get_recall(indices, targets): #recall --> wether next item in session is within top K recommended items or not
   
    if len(targets.size()) == 1:
        # One hot label branch
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall
    else:        
        # Multi hot label branch
        recall = []
        for preds, gt in zip(indices, targets):            
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)
            success = (preds_binary * gt_binary).sum() > 0
            recall.append(int(success))        
        return torch.Tensor(recall).float().mean()
    











