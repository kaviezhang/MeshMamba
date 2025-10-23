import copy
import os
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import math
import numpy as np
from data.MeshRotateDataset import MeshDataset
from models import *
from utils.retrival import append_feature, calculate_map
from utils.loss_function import *
import torch.nn.init as init
import yaml
import datetime
import sys
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def train_model(model, criterion, optimizer, scheduler, cfg):
    best_acc = np.Inf
    best_map = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg['max_epoch']):

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
        print('-' * 60)

        # adjust_learning_rate(cfg, epoch, optimizer)
        for phrase in ['train', 'test']:

            if phrase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            test_loss = 0.0
            total_cc = 0.0
            total_sim = 0.0
            total_kld = 0.0
            total_nss = 0.0
            ft_all, lbl_all = None, None
            name_list = []

            for i, collated_dict in enumerate(data_loader[phrase]):
                centers = collated_dict['centers'].permute(0, 2, 1)
                normals = collated_dict['normals'].permute(0, 2, 1)
                corners = collated_dict['corners'].permute(0, 2, 1)
                faces = collated_dict['faces'].cuda()
                verts = collated_dict['verts'].cuda()
                corners = corners.cuda()
                neighbor_index = collated_dict['neighbors'].cuda()
                ring_1 = collated_dict['ring_1'].cuda()
                ring_2 = collated_dict['ring_2'].cuda()
                ring_3 = collated_dict['ring_3'].cuda()
                targets = collated_dict['target'].cuda()
                texture = collated_dict['texture'].cuda()
                uv_grid = collated_dict['uv_grid'].cuda()
                centers = centers.cuda()
                normals = normals.cuda()
                mesh_name = collated_dict['mesh_name']
                with torch.set_grad_enabled(phrase == 'train'):
                    outputs = model(verts=verts,
                                    faces=faces,
                                    centers=centers,
                                    normals=normals,
                                    corners=corners,
                                    neighbor_index=neighbor_index,
                                    ring_1=ring_1,
                                    ring_2=ring_2,
                                    ring_3=ring_3,
                                    face_colors=0,
                                    face_textures=0,
                                    texture=texture,
                                    uv_grid=uv_grid)
                    loss_cc = loss_fn(outputs, targets, loss_type='cc')
                    loss_sim = loss_fn(outputs, targets, loss_type='sim')
                    loss_kldiv = loss_fn(outputs, targets, loss_type='kldiv')
                    loss_l1 = criterion(outputs, targets)
                    loss = loss_l1 * 1.0 + loss_kldiv * 0.2

                    if phrase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if phrase == 'test' and cfg['retrieval_on']:
                        ft_all = append_feature(ft_all, outputs.detach().cpu())
                        lbl_all = append_feature(lbl_all, targets.detach().cpu())
                        name_list = name_list + mesh_name

                    # statistics
                    running_loss += loss.item() * centers.size(0)
                    total_cc += loss_cc.item() * centers.size(0)
                    total_sim += loss_sim.item() * centers.size(0)
                    total_kld += loss_kldiv.item() * centers.size(0)

            epoch_loss = running_loss / len(data_set[phrase])
            total_cc = total_cc / len(data_set[phrase])
            total_sim = total_sim / len(data_set[phrase])
            total_kld = total_kld / len(data_set[phrase])
            total_nss = 0.0
            print('Average {},cc={},sim={},kld={},nss={}'.format(phrase, total_cc, total_sim, total_kld, total_nss))

            if phrase == 'train':
                print('{} Loss: {:.4f}'.format(phrase, epoch_loss))
                scheduler.step()

            if phrase == 'test':
                mse_loss = np.mean((ft_all - lbl_all) ** 2)
                test_loss = epoch_loss
                if test_loss < best_acc:
                    best_acc = test_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # 保存预测结果
                    print_info = ('{} Loss: {:.4f} MES_Loss: {:.4f} (best {:.4f})'
                                  .format(phrase, epoch_loss, mse_loss, best_acc))
                    print(print_info)
                    for index, _ in enumerate(name_list):
                        name = visual_path + name_list[index] + ".csv"
                        np.savetxt(name, ft_all[index], fmt='%f', delimiter=",")

                if epoch % cfg['save_steps'] == 0:
                    torch.save(copy.deepcopy(model.state_dict()),
                               os.path.join(cfg['ckpt_root'], '{}.pkl'.format(epoch)))

    print('Exp {} Best val mse: {:.4f}'.format(exp_id, best_acc))
    print('Config: {}'.format(cfg))

    return best_model_wts


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # 使用Xavier初始化权重
            init.xavier_uniform_(m.weight)
            # 初始化偏置为零
            if m.bias is not None:
                init.constant_(m.bias, 0)


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if __name__ == '__main__':
    # Training Parameters
    SHUFFLE = False

    with open("config/MeshSaliency.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
    # criterion
    criterion = nn.L1Loss()
    loss_fn = SaliencyLoss()

    script_name = os.path.basename(__file__)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = "log/" + str(current_time)
    os.mkdir(file_path)

    # loop the exp_id
    for exp_id in range(10):
        current_log = Logger(filename="{}/{}.log".format(file_path, exp_id))
        sys.stdout = current_log
        print(script_name)

        paths_dataset = np.loadtxt("paths_dataset_rgb.txt", dtype=str, delimiter=",")
        dataset_paths = {}
        if SHUFFLE:
            np.random.shuffle(paths_dataset)
            np.savetxt("{}/path_{}.txt".format(file_path, exp_id), paths_dataset, fmt="%s", delimiter=",")
        dataset_paths["train"] = paths_dataset[:int(len(paths_dataset) * 0.8)]
        dataset_paths["test"] = paths_dataset[int(len(paths_dataset) * 0.8):]

        # dataset
        data_set = {
            x: MeshDataset(cfg=cfg['dataset'], part=x, mesh_paths=dataset_paths[x]) for x in ['train', 'test']
        }
        data_loader = {
            x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'], num_workers=0, shuffle=True, pin_memory=False)
            for x in ['train', 'test']
        }

        # build the visualization path
        visual_path = "{}/visualization_{}/".format(file_path, exp_id)
        if not os.path.exists(visual_path):
            os.mkdir(visual_path)

        # prepare model
        model = MambaMesh(cfg['mamba']).cuda()
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        initialize_weights(model)
        device_ids = list(range(torch.cuda.device_count()))
        # model = nn.DataParallel(model.cuda(), device_ids=device_ids)

        # optimizer
        if cfg['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                                  weight_decay=cfg['weight_decay'])
        elif cfg['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        # scheduler
        if cfg['scheduler'] == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'])
        elif cfg['scheduler'] == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['max_epoch'])
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        # start training
        if not os.path.exists(cfg['ckpt_root']):
            os.mkdir(cfg['ckpt_root'])
        best_model_wts = train_model(model, criterion, optimizer, scheduler, cfg)
        torch.save(best_model_wts, os.path.join(cfg['ckpt_root'], 'MeshSal_best.pkl'))
        current_log.close()
