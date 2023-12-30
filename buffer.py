import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from epoch import epoch, epoch_test, itm_eval
import copy
import wandb
import warnings
import datetime
from data import get_dataset_flickr, textprocess, textprocess_train
from networks import CLIPModel_full
from utils import get_dataset, TensorDataset, load_or_process_file
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    #wandb.init(mode="disabled")
    wandb.init(project='DatasetDistillation', entity='dataset_distillation', config=args, name=args.name)


    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.distributed = torch.cuda.device_count() > 1

    
    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.image_encoder, args.text_encoder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ''' organize the datasets '''
    trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)
    # data = np.load(f'{args.dataset}_{args.text_encoder}_text_embed.npz')
    data = load_or_process_file('text', textprocess, args, testloader)
    bert_test_embed = torch.from_numpy(data['bert_test_embed']).cpu()


    img_trajectories = []
    txt_trajectories = []

    for it in range(0, args.num_experts):

        ''' Train synthetic data '''
        
        teacher_net = CLIPModel_full(args)
        img_teacher_net = teacher_net.image_encoder.to(args.device)
        txt_teacher_net = teacher_net.text_projection.to(args.device)
        if args.text_trainable:
            txt_teacher_net = teacher_net.text_encoder.to(args.device)
        if args.distributed:
            img_teacher_net = torch.nn.DataParallel(img_teacher_net)
            txt_teacher_net = torch.nn.DataParallel(txt_teacher_net)
        img_teacher_net.train()
        txt_teacher_net.train()
        lr_img = args.lr_teacher_img
        lr_txt = args.lr_teacher_txt

        teacher_optim_img = torch.optim.SGD(img_teacher_net.parameters(), lr=lr_img, momentum=args.mom, weight_decay=args.l2) 
        teacher_optim_txt = torch.optim.SGD(txt_teacher_net.parameters(), lr=lr_txt, momentum=args.mom, weight_decay=args.l2) 
        teacher_optim_img.zero_grad()
        teacher_optim_txt.zero_grad()

        img_timestamps = []
        txt_timestamps = []

        img_timestamps.append([p.detach().cpu() for p in img_teacher_net.parameters()])
        txt_timestamps.append([p.detach().cpu() for p in txt_teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]

        for e in range(args.train_epochs):
            train_loss, train_acc = epoch(e, trainloader, teacher_net, teacher_optim_img, teacher_optim_txt, args)
            score_val_i2t, score_val_t2i = epoch_test(testloader, teacher_net, args.device, bert_test_embed)
            val_result = itm_eval(score_val_i2t, score_val_t2i, testloader.dataset.txt2img, testloader.dataset.img2txt)  
            
        
            wandb.log({"train_loss": train_loss})
            wandb.log({"train_acc": train_acc})
            wandb.log({"txt_r1": val_result['txt_r1']})
            wandb.log({"txt_r5": val_result['txt_r5']})
            wandb.log({"txt_r10": val_result['txt_r10']})
            wandb.log({"txt_r_mean": val_result['txt_r_mean']})
            wandb.log({"img_r1": val_result['img_r1']})
            wandb.log({"img_r5": val_result['img_r5']})
            wandb.log({"img_r10": val_result['img_r10']})
            wandb.log({"img_r_mean": val_result['img_r_mean']})
            wandb.log({"r_mean": val_result['r_mean']})

            print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tImg R@1: {}\tR@5: {}\tR@10: {}\tR@Mean: {}\tTxt R@1: {}\tR@5: {}\tR@10: {}\tR@Mean: {}".format(
                it, e, train_acc,
                val_result['img_r1'], val_result['img_r5'], val_result['img_r10'], val_result['img_r_mean'], 
                val_result['txt_r1'], val_result['txt_r5'], val_result['txt_r10'], val_result['txt_r_mean'])) 
            img_timestamps.append([p.detach().cpu() for p in img_teacher_net.parameters()])
            txt_timestamps.append([p.detach().cpu() for p in txt_teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim_img = torch.optim.SGD(img_teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim_txt = torch.optim.SGD(txt_teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim_img.zero_grad()
                teacher_optim_txt.zero_grad()

        img_trajectories.append(img_timestamps)
        txt_trajectories.append(txt_timestamps)
        n = 0
        while os.path.exists(os.path.join(save_dir, "img_replay_buffer_{}.pt".format(n))):
            n += 1
        print("Saving {}".format(os.path.join(save_dir, "img_replay_buffer_{}.pt".format(n))))
        torch.save(img_trajectories, os.path.join(save_dir, "img_replay_buffer_{}.pt".format(n)))
        print("Saving {}".format(os.path.join(save_dir, "txt_replay_buffer_{}.pt".format(n))))
        torch.save(txt_trajectories, os.path.join(save_dir, "txt_replay_buffer_{}.pt".format(n)))

        img_trajectories = []
        txt_trajectories = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='flickr', choices=['flickr', 'coco'], help='dataset')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='./data/Flickr30k/', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization') 
    parser.add_argument('--save_interval', type=int, default=10)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument('--name', type=str, default=current_time, help='name of wandb run')
    parser.add_argument('--text_pretrained', type=bool, default=True, help='text_pretrained')
    parser.add_argument('--image_pretrained', type=bool, default=True, help='image_pretrained')
    parser.add_argument('--text_trainable', type=bool, default=False, help='text_trainable')
    parser.add_argument('--image_trainable', type=bool, default=True, help='image_trainable') 
    parser.add_argument('--batch_size_train', type=int, default=128, help='batch_size_train')
    parser.add_argument('--batch_size_test', type=int, default=128, help='batch_size_test')
    parser.add_argument('--image_root', type=str, default='./Flickr30k/flickr-image-dataset/flickr30k-images/', help='location of image root')
    parser.add_argument('--ann_root', type=str, default='./Flickr30k/ann_file/', help='location of ann root')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--k_test', type=int, default=128, help='k_test')
    parser.add_argument('--load_npy', type=bool, default=False, help='load_npy')
    parser.add_argument('--image_encoder', type=str, default='resnet50', choices=['nfnet', 'resnet18_gn', 'vit_tiny', 'nf_resnet50', 'nf_regnet'],  help='image encoder')
    parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 'clip'], help='text encoder')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--measure', default='cosine',
                    help='Similarity measure used (cosine|order)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--only_has_image_projection', type=bool, default=False, help='None')
    parser.add_argument('--grounding', type=bool, default=False, help='None')
    parser.add_argument('--distill', type=bool, default=False, help='if distill')
    args = parser.parse_args()

    main(args)

