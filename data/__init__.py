from torchvision import transforms
from transform.randaugment import RandomAugment
from torchvision.transforms.functional import InterpolationMode
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import json
from PIL import Image
import os
from torchvision import transforms as T
from networks import CLIPModel_full
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.coco_dataset import coco_train, coco_caption_eval, coco_retrieval_eval
import numpy as np
from tqdm import tqdm
@torch.no_grad()
def textprocess(args, testloader):
    net = CLIPModel_full(args).to('cuda')
    net.eval() 
    texts = testloader.dataset.text 
    if args.dataset in ['flickr', 'coco']:
        if args.dataset == 'flickr':
            bert_test_embed = net.text_encoder(texts)
        elif args.dataset == 'coco':
            bert_test_embed = torch.cat((net.text_encoder(texts[:10000]), net.text_encoder(texts[10000:20000]), net.text_encoder(texts[20000:])), dim=0)

        bert_test_embed_np = bert_test_embed.cpu().numpy()
        np.savez(f'{args.dataset}_{args.text_encoder}_text_embed.npz', bert_test_embed=bert_test_embed_np) 
    else:
        raise NotImplementedError
    return 

@torch.no_grad()
def textprocess_train(args, texts):
    net = CLIPModel_full(args).to('cuda')
    net.eval() 
    chunk_size = 2000
    chunks = []
    for i in tqdm(range(0, len(texts), chunk_size)):
        chunk = net.text_encoder(texts[i:i + chunk_size]).cpu()
        chunks.append(chunk)
        del chunk
        torch.cuda.empty_cache()  # free up memory
    bert_test_embed = torch.cat(chunks, dim=0)

    print('bert_test_embed.shape: ', bert_test_embed.shape)
    bert_test_embed_np = bert_test_embed.numpy()
    if args.dataset in ['flickr', 'coco']:
        np.savez(f'{args.dataset}_{args.text_encoder}_train_text_embed.npz', bert_test_embed=bert_test_embed_np) 
    else:
        raise NotImplementedError
    return 


def create_dataset(args, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(args.image_size,scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    if args.dataset=='flickr':          
        train_dataset = flickr30k_train(transform_train, args.image_root, args.ann_root)
        val_dataset = flickr30k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val') 
        test_dataset = flickr30k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')         
        return train_dataset, val_dataset, test_dataset    
    
    elif args.dataset=='coco':             
        train_dataset = coco_train(transform_train, args.image_root, args.ann_root)
        val_dataset = coco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val') 
        test_dataset = coco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')         
        return train_dataset, val_dataset, test_dataset     
    else: 
        raise NotImplementedError
    return train_dataset, val_dataset, test_dataset     

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    


def get_dataset_flickr(args):
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(args)  

    samplers = [None, None, None]
    train_shuffle = True
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[args.batch_size_train]+[args.batch_size_test]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[train_shuffle, False, False], 
                                                          collate_fns=[None,None,None])  
    return train_loader, test_loader, train_dataset, test_dataset

