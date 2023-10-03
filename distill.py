import argparse
import copy
import datetime
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import math

from transformers import BertTokenizer, BertConfig, BertModel
import wandb

from data import get_dataset_flickr, textprocess, textprocess_train
from epoch import evaluate_synset, epoch, epoch_test, itm_eval
from networks import CLIPModel_full, TextEncoder
from reparam_module import ReparamModule
from utils import DiffAugment, ParamDiffAug, TensorDataset, get_dataset, get_network, get_eval_pool, get_time


def shuffle_files(img_expert_files, txt_expert_files):
    # Check if both lists have the same length and if the lists are not empty
    assert len(img_expert_files) == len(txt_expert_files), "Number of image files and text files does not match"
    assert len(img_expert_files) != 0, "No files to shuffle"
    shuffled_indices = np.random.permutation(len(img_expert_files))

    # Apply the shuffled indices to both lists
    img_expert_files = np.take(img_expert_files, shuffled_indices)
    txt_expert_files = np.take(txt_expert_files, shuffled_indices)
    print(f"img_expert_files: {img_expert_files}")
    print(f"txt_expert_files: {txt_expert_files}")
    return img_expert_files, txt_expert_files

def nearest_neighbor(sentences, query_embeddings, database_embeddings):
    """Find the nearest neighbors for a batch of embeddings.

    Args:
    sentences: The original sentences from which the embeddings were computed.
    query_embeddings: A batch of embeddings for which to find the nearest neighbors.
    database_embeddings: All pre-computed embeddings.

    Returns:
    A list of the most similar sentences for each embedding in the batch.
    """
    nearest_neighbors = []
    
    for query in query_embeddings:
        similarities = cosine_similarity(query.reshape(1, -1), database_embeddings)
        
        most_similar_index = np.argmax(similarities)
        
        nearest_neighbors.append(sentences[most_similar_index])
        
    return nearest_neighbors


def get_images_texts(n, dataset):
    """Get random n images and corresponding texts from the dataset.

    Args:
    n: Number of images and texts to retrieve.
    dataset: The dataset containing image-text pairs.

    Returns:
    A tuple containing two elements:
      - A tensor of randomly selected images.
      - A tensor of the corresponding texts, encoded as floats.
    """
    # Generate n unique random indices
    idx_shuffle = np.random.permutation(len(dataset))[:n]

    # Initialize the text encoder
    text_encoder = TextEncoder(args)

    image_syn = torch.stack([dataset[i][0] for i in idx_shuffle])
    text_syn = text_encoder([dataset[i][1] for i in idx_shuffle], device="cpu")

    return image_syn, text_syn.float()


def load_or_process_file(file_type, process_func, args, data_source):
    """
    Load the processed file if it exists, otherwise process the data source and create the file.

    Args:
    file_type: The type of the file (e.g., 'train', 'test').
    process_func: The function to process the data source.
    args: The arguments required by the process function and to build the filename.
    data_source: The source data to be processed.

    Returns:
    The loaded data from the file.
    """
    filename = f'{args.dataset}_{args.text_encoder}_{file_type}_embed.npz'


    if not os.path.exists(filename):
        print(f'Creating {filename}')
        process_func(args, data_source)
    else:
        print(f'Loading {filename}')
    
    return np.load(filename)


def main(args):  
    ''' organize the real train dataset '''  
    trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)

    train_sentences = train_dataset.get_all_captions() 

    data = load_or_process_file('text', textprocess, args, testloader)
    train_caption = load_or_process_file('train_text', textprocess_train, args, train_sentences)

    bert_test_embed = torch.from_numpy(data['bert_test_embed']).cpu()
    print("The shape of bert_test_embed: {}".format(bert_test_embed.shape))
    train_caption_embed = torch.from_numpy(train_caption['bert_test_embed']).cpu()
    print("The shape of train_caption_embed: {}".format(train_caption_embed.shape))

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()

    if args.dsa:
        args.dc_aug_param = None

    # wandb.init(mode="disabled")
    wandb.init(project='DatasetDistillation', entity='dataset_distillation', config=args, name=args.name)
    
    args.dsa_param = ParamDiffAug()
    zca_trans = args.zca_trans if args.zca else None
    args.zca_trans = zca_trans
    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    syn_lr_img = torch.tensor(args.lr_teacher_img).to(args.device)
    syn_lr_txt = torch.tensor(args.lr_teacher_txt).to(args.device)

    ''' initialize the synthetic data '''
    image_syn, text_syn = get_images_texts(args.num_queries, train_dataset)
    
    if args.pix_init == 'noise':
        mean = torch.tensor([-0.0626, -0.0221,  0.0680])
        std = torch.tensor([1.0451, 1.0752, 1.0539])
        image_syn = torch.randn([args.num_queries, 3, 224, 224])
        for c in range(3):
            image_syn[:, c] = image_syn[:, c] * std[c] + mean[c]
        print('Initialized synthetic image from random noise')

    if args.txt_init == 'noise':
        text_syn = torch.normal(mean=-0.0094, std=0.5253, size=(args.num_queries, 768))
        print('Initialized synthetic text from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_img.zero_grad()

    syn_lr_img = syn_lr_img.to(args.device).requires_grad_(True)
    syn_lr_txt = syn_lr_txt.to(args.device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr_img, syn_lr_txt], lr=args.lr_lr, momentum=0.5)
    
    text_syn = text_syn.detach().to(args.device).requires_grad_(True)
    optimizer_txt = torch.optim.SGD([text_syn], lr=args.lr_txt, momentum=0.5)
    optimizer_txt.zero_grad()
    sentence_list = nearest_neighbor(train_sentences, text_syn.detach().cpu(), train_caption_embed)
    if args.draw:
        wandb.log({"original_sentence_list": wandb.Html('<br>'.join(sentence_list))})
        wandb.log({"original_synthetic_images": wandb.Image(torch.nan_to_num(image_syn.detach().cpu()))})

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    expert_dir = args.buffer_path
    print("Expert Dir: {}".format(expert_dir))


    img_expert_files = []
    txt_expert_files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "img_replay_buffer_{}.pt".format(n))):
        img_expert_files.append(os.path.join(expert_dir, "img_replay_buffer_{}.pt".format(n)))
        txt_expert_files.append(os.path.join(expert_dir, "txt_replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    
    img_expert_files, txt_expert_files = shuffle_files(img_expert_files, txt_expert_files)
    
    file_idx = 0
    expert_idx = 0
    print("loading file {}".format(img_expert_files[file_idx]))
    print("loading file {}".format(txt_expert_files[file_idx]))
    
    img_buffer = torch.load(img_expert_files[file_idx])
    txt_buffer = torch.load(txt_expert_files[file_idx])

    for it in tqdm(range(args.Iteration + 1)):
        save_this_it = True

        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            print('-------------------------\nEvaluation\nimage_model_train = %s, text_model_train = %s, iteration = %d'%(args.image_encoder, args.text_encoder, it))
            if args.dsa:
                print('DSA augmentation strategy: \n', args.dsa_strategy)
                print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
            else:
                print('DC augmentation parameters: \n', args.dc_aug_param)

            accs_train = []
            img_r1s = []
            img_r5s = []
            img_r10s = []
            img_r_means = []

            txt_r1s = []
            txt_r5s = []
            txt_r10s = []
            txt_r_means = []

            r_means = []
            for it_eval in range(args.num_eval):
                net_eval = CLIPModel_full(args, eval_stage=args.transfer)

                with torch.no_grad():
                    image_save = image_syn
                    text_save = text_syn
                image_syn_eval, text_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(text_save.detach()) # avoid any unaware modification

                args.lr_net = syn_lr_img.item()
                print(image_syn_eval.shape)
                _, acc_train, val_result = evaluate_synset(it_eval, net_eval, image_syn_eval, text_syn_eval, testloader, args, bert_test_embed)
                print('Evaluate_%02d: Img R@1 = %.4f, Img R@5 = %.4f, Img R@10 = %.4f, Img R@Mean = %.4f, Txt R@1 = %.4f, Txt R@5 = %.4f, Txt R@10 = %.4f, Txt R@Mean = %.4f, R@Mean = %.4f' % 
                    (it_eval, 
                    val_result['img_r1'], val_result['img_r5'], val_result['img_r10'], val_result['img_r_mean'], 
                    val_result['txt_r1'], val_result['txt_r5'], val_result['txt_r10'], val_result['txt_r_mean'], 
                    val_result['r_mean']))

                img_r1s.append(val_result['img_r1'])
                img_r5s.append(val_result['img_r5'])
                img_r10s.append(val_result['img_r10'])
                img_r_means.append(val_result['img_r_mean'])
                
                txt_r1s.append(val_result['txt_r1'])
                txt_r5s.append(val_result['txt_r5'])
                txt_r10s.append(val_result['txt_r10'])
                txt_r_means.append(val_result['txt_r_mean'])
                r_means.append(val_result['r_mean'])
                
                if not args.std:
                    wandb.log({"txt_r1": val_result['txt_r1']})
                    wandb.log({"txt_r5": val_result['txt_r5']})
                    wandb.log({"txt_r10": val_result['txt_r10']})
                    wandb.log({"txt_r_mean": val_result['txt_r_mean']})
                    wandb.log({"img_r1": val_result['img_r1']})
                    wandb.log({"img_r5": val_result['img_r5']})
                    wandb.log({"img_r10": val_result['img_r10']})
                    wandb.log({"img_r_mean": val_result['img_r_mean']})
                    wandb.log({"r_mean": val_result['r_mean']})
            if args.std:
                img_r1_mean, img_r1_std = np.mean(img_r1s), np.std(img_r1s)
                img_r5_mean, img_r5_std = np.mean(img_r5s), np.std(img_r5s)
                img_r10_mean, img_r10_std = np.mean(img_r10s), np.std(img_r10s)
                img_r_mean_mean, img_r_mean_std = np.mean(img_r_means), np.std(img_r_means)

                txt_r1_mean, txt_r1_std = np.mean(txt_r1s), np.std(txt_r1s)
                txt_r5_mean, txt_r5_std = np.mean(txt_r5s), np.std(txt_r5s)
                txt_r10_mean, txt_r10_std = np.mean(txt_r10s), np.std(txt_r10s)
                txt_r_mean_mean, txt_r_mean_std = np.mean(txt_r_means), np.std(txt_r_means)
                r_mean_mean, r_mean_std = np.mean(r_means), np.std(r_means)

                wandb.log({'Mean/txt_r1': txt_r1_mean, 'Std/txt_r1': txt_r1_std})
                wandb.log({'Mean/txt_r5': txt_r5_mean, 'Std/txt_r5': txt_r5_std})
                wandb.log({'Mean/txt_r10': txt_r10_mean, 'Std/txt_r10': txt_r10_std})
                wandb.log({'Mean/txt_r_mean': txt_r_mean_mean, 'Std/txt_r_mean': txt_r_mean_std})
                wandb.log({'Mean/img_r1': img_r1_mean, 'Std/img_r1': img_r1_std})
                wandb.log({'Mean/img_r5': img_r5_mean, 'Std/img_r5': img_r5_std})
                wandb.log({'Mean/img_r10': img_r10_mean, 'Std/img_r10': img_r10_std})
                wandb.log({'Mean/img_r_mean': img_r_mean_mean, 'Std/img_r_mean': img_r_mean_std})
                wandb.log({'Mean/r_mean': r_mean_mean, 'Std/r_mean': r_mean_std})

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            if args.draw:
                with torch.no_grad():
                    image_save = image_syn_eval.cuda()
                    text_save = text_syn_eval.cuda()
                    save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)
                    print("Saving to {}".format(save_dir))

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    #torch.save(image_save, os.path.join(save_dir, "images_{}.pt".format(it)))
                    #torch.save(text_save, os.path.join(save_dir, "labels_{}.pt".format(it)))

                    #torch.save(image_save, os.path.join(save_dir, "images_best.pt".format(it)))
                    #torch.save(text_save, os.path.join(save_dir, "labels_best.pt".format(it)))

                    wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)  # Move tensor to CPU before converting to NumPy

                    if args.ipc < 50 or args.force_save:
                        upsampled = image_save[:90]
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        sentence_list = nearest_neighbor(train_sentences, text_syn.cpu(), train_caption_embed)
                        sentence_list = sentence_list[:90]
                        torchvision.utils.save_image(grid, os.path.join(save_dir, "synthetic_images_{}.png".format(it)))
                        
                        with open(os.path.join(save_dir, "synthetic_sentences_{}.txt".format(it)), "w") as file:
                            file.write('\n'.join(sentence_list))
                        wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it) 
                        wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it) 
                        wandb.log({"Synthetic_Sentences": wandb.Html('<br>'.join(sentence_list))}, step=it)
                        print("finish saving images")

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std).cpu()  # Move to CPU
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled[:90], nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid))}, step=it)
                            torchvision.utils.save_image(grid, os.path.join(save_dir, "clipped_synthetic_images_{}_std_{}.png".format(it, clip_val)))
                            

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save.cpu())  # Move to CPU for ZCA transformation
                        torch.save(image_save, os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid))}, step=it)  # Log GPU tensor directly
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        wandb.log({"Synthetic_LR_Image": syn_lr_img.detach().cpu()}, step=it)
        wandb.log({"Synthetic_LR_Text": syn_lr_txt.detach().cpu()}, step=it)

        torch.cuda.empty_cache()
        student_net = CLIPModel_full(args)
        img_student_net = ReparamModule(student_net.image_encoder.to('cpu')).to('cuda')
        txt_student_net = ReparamModule(student_net.text_projection.to('cpu')).to('cuda')

        if args.distributed:
            img_student_net = torch.nn.DataParallel(img_student_net)
            txt_student_net = torch.nn.DataParallel(txt_student_net)

        img_student_net.train()
        txt_student_net.train()
        img_num_params = sum([np.prod(p.size()) for p in (img_student_net.parameters())])
        txt_num_params = sum([np.prod(p.size()) for p in (txt_student_net.parameters())])


        img_expert_trajectory = img_buffer[expert_idx]
        txt_expert_trajectory = txt_buffer[expert_idx]
        expert_idx += 1
        if expert_idx == len(img_buffer):
            expert_idx = 0
            file_idx += 1
            if file_idx == len(img_expert_files): 
                file_idx = 0
                img_expert_files, txt_expert_files = shuffle_files(img_expert_files, txt_expert_files)
            print("loading file {}".format(img_expert_files[file_idx]))
            print("loading file {}".format(txt_expert_files[file_idx]))
            if args.max_files != 1:
                del img_buffer
                del txt_buffer
                img_buffer = torch.load(img_expert_files[file_idx])
                txt_buffer = torch.load(txt_expert_files[file_idx])

        start_epoch = np.random.randint(0, args.max_start_epoch)
        img_starting_params = img_expert_trajectory[start_epoch]
        txt_starting_params = txt_expert_trajectory[start_epoch]

        img_target_params = img_expert_trajectory[start_epoch+args.expert_epochs]
        txt_target_params = txt_expert_trajectory[start_epoch+args.expert_epochs]

        img_target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in img_target_params], 0)
        txt_target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in txt_target_params], 0)

        img_student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in img_starting_params], 0).requires_grad_(True)]
        txt_student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in txt_starting_params], 0).requires_grad_(True)]

        img_starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in img_starting_params], 0)
        txt_starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in txt_starting_params], 0)
        syn_images = image_syn
        syn_texts = text_syn

        img_param_loss_list = []
        txt_param_loss_list = []

        img_param_dist_list = []
        txt_param_dist_list = []

        indices_chunks = []
        for step in range(args.syn_steps): 
            indices = torch.randperm(len(syn_images))
            these_indices = indices[:args.mini_batch_size]
            #these_indices = indices
            x = syn_images[these_indices]
            this_y = syn_texts[these_indices]
            if args.distributed:
                img_forward_params = img_student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                txt_forward_params = txt_student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                img_forward_params = img_student_params[-1]
                txt_forward_params = txt_student_params[-1]

            x = img_student_net(x, flat_param=img_forward_params)
            x = x / x.norm(dim=1, keepdim=True)
            this_y = txt_student_net(this_y, flat_param=txt_forward_params)
            this_y = this_y / this_y.norm(dim=1, keepdim=True)
            image_logits = logit_scale * x.float() @ this_y.float().t() 
            ground_truth = torch.arange(len(image_logits)).type_as(image_logits).long()
            contrastive_loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            
            img_grad = torch.autograd.grad(contrastive_loss, img_student_params[-1], create_graph=True)[0]
            txt_grad = torch.autograd.grad(contrastive_loss, txt_student_params[-1], create_graph=True)[0]
            print(contrastive_loss)
            img_student_params.append(img_student_params[-1] - syn_lr_img * img_grad) 
            txt_student_params.append(txt_student_params[-1] - syn_lr_txt * txt_grad)
        img_param_loss = torch.tensor(0.0).to(args.device)
        img_param_dist = torch.tensor(0.0).to(args.device)
        txt_param_loss = torch.tensor(0.0).to(args.device)
        txt_param_dist = torch.tensor(0.0).to(args.device)


        img_param_loss += torch.nn.functional.mse_loss(img_student_params[-1], img_target_params, reduction="sum")
        img_param_dist += torch.nn.functional.mse_loss(img_starting_params, img_target_params, reduction="sum")
        txt_param_loss += torch.nn.functional.mse_loss(txt_student_params[-1], txt_target_params, reduction="sum")
        txt_param_dist += torch.nn.functional.mse_loss(txt_starting_params, txt_target_params, reduction="sum")

        img_param_loss_list.append(img_param_loss)
        img_param_dist_list.append(img_param_dist)
        txt_param_loss_list.append(txt_param_loss)
        txt_param_dist_list.append(txt_param_dist)
        

        img_param_loss /= img_param_dist
        txt_param_loss /= txt_param_dist
        grand_loss = img_param_loss + txt_param_loss

        if math.isnan(img_param_loss):
            break
        print("img_param_loss: {}".format(img_param_loss))
        print("txt_param_loss: {}".format(txt_param_loss))

        optimizer_lr.zero_grad()
        optimizer_img.zero_grad()
        optimizer_txt.zero_grad()
        
        grand_loss.backward()
        # clip_value = 0.5
        
        #torch.nn.utils.clip_grad_norm_([image_syn], clip_value)
        #torch.nn.utils.clip_grad_norm_([text_syn], clip_value)
        #torch.nn.utils.clip_grad_norm_([syn_lr_img], clip_value)
        #torch.nn.utils.clip_grad_norm_([syn_lr_txt], clip_value)
        print("syn_lr_img: {}".format(syn_lr_img.grad))
        print("syn_lr_txt: {}".format(syn_lr_txt.grad))
        wandb.log({"syn_lr_img": syn_lr_img.grad.detach().cpu()}, step=it)
        wandb.log({"syn_lr_txt": syn_lr_txt.grad.detach().cpu()}, step=it)

        optimizer_lr.step()
        optimizer_img.step()
        optimizer_txt.step()

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in img_student_params:
            del _
        for _ in txt_student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='flickr30k', help='dataset')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=50, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=50, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=50000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_txt', type=float, default=1000, help='learning rate for updating synthetic texts')
    parser.add_argument('--lr_lr', type=float, default=1e-03, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='learning rate for updating network parameters')
    
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--txt_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic texts from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='./data/Flickr30k/', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument('--name', type=str, default=current_time, help='name of wandb run')
    parser.add_argument('--num_queries', type=int, default=100, help='number of queries')
    parser.add_argument('--mini_batch_size', type=int, default=100, help='number of queries')
    parser.add_argument('--basis', type=bool, default=False, help='whether use basis or not')
    parser.add_argument('--n_basis', type=int, default=64, help='n_basis')
    parser.add_argument('--recursive', type=bool, default=False, help='whether use basis or not')
    parser.add_argument('--load_npy', type=bool, default=False, help='load_npy')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--image_root', type=str, default='./Flickr30k/flickr-image-dataset/flickr30k-images/', help='location of image root')
    parser.add_argument('--ann_root', type=str, default='./Flickr30k/ann_file/', help='location of ann root')
    parser.add_argument('--batch_size_train', type=int, default=128, help='batch_size_train')
    parser.add_argument('--batch_size_test', type=int, default=128, help='batch_size_test')
    parser.add_argument('--image_encoder', type=str, default='nfnet', choices=['clip', 'nfnet', 'vit', 'nf_resnet50'],  help='image encoder')
    parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 'clip'], help='text encoder')
    parser.add_argument('--text_pretrained', type=bool, default=True, help='text_pretrained')
    parser.add_argument('--image_pretrained', type=bool, default=True, help='image_pretrained')
    parser.add_argument('--text_trainable', type=bool, default=False, help='text_trainable')
    parser.add_argument('--image_trainable', type=bool, default=True, help='image_trainable') 
    parser.add_argument('--only_has_image_projection', type=bool, default=False, help='None')
    parser.add_argument('--distill', type=bool, default=True, help='whether distill')
    parser.add_argument('--optimize', type=str, default='reparam', choices=['reparam', 'ift'], help='matching_train')
    parser.add_argument('--image_only', type=bool, default=False, help='None')
    parser.add_argument('--text_only', type=bool, default=False, help='None')
    parser.add_argument('--draw', type=bool, default=True, help='None')
    parser.add_argument('--transfer', type=bool, default=False, help='transfer cross architecture')
    parser.add_argument('--std', type=bool, default=False, help='standard deviation')
    args = parser.parse_args()

    main(args)