import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.facenet import Facenet
from nets.facenet_training import (get_lr_scheduler, set_optimizer_lr,
                                   triplet_loss, weights_init)
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate
from utils.utils import get_num_classes, show_config
from utils.utils_fit import fit_one_epoch

from torch.utils.tensorboard import SummaryWriter



if __name__ == "__main__":
    #   tensorboard调用
    writer = SummaryWriter(log_dir='/root/tf-logs',flush_secs=60)
    Cuda            = True
    distributed     = False
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    sync_bn         = False
    fp16            = False
    #   指向根目录下的cls_train.txt，读取人脸路径与标签
    annotation_path = "cls_train.txt"
    #   输入网络图像大小，常用设置如[112, 112, 3]
    input_shape     = [160, 160, 3]
    backbone        = "mobilenet"
    model_path      ='model_data/facenet_mobilenet.pth'
    pretrained      = False

    '''
           Adam:
               Init_Epoch = 0, Epoch = 100, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0
           SGD:
               Init_Epoch = 0, Epoch = 100, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4
    '''
    batch_size      = 126   # batch size have to be divided by 3
    Init_Epoch      = 0
    Epoch           = 100

    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    
    #   optimizer_type  adam、sgd
    #                   Adam  Init_lr=1e-3
    #                   SGD   Init_lr=1e-2

    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0

    
    lr_decay_type       = "cos" #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    save_period         = 1
    #   save_dir        权值与日志文件保存的文件夹-#
    save_dir            = 'logs'
    num_workers     = 4
    lfw_eval_flag   = True
    lfw_dir_path    = "lfw"
    lfw_pairs_path  = "model_data/lfw_pair.txt"

    #   设置用到的显卡
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    num_classes = get_num_classes(annotation_path)
    #   载入模型，加载训练权重
    model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        #   根据预训练权重的Key和模型的Key进行加载

        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        #   显示没有匹配上的Key
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    loss            = triplet_loss()
    #   record Loss

    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=32, shuffle=False) if lfw_eval_flag else None

    #   由于主要是用LFW进行验证，WebFace数据集可以取0.01用于验证，0.99用于训练
    val_split = 0.01
    with open(annotation_path,"r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    show_config(
        num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    if True:
        if batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3.")

        #   判断当前batch_size，自适应调整学习率
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        #   根据optimizer_type选择优化器
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #   学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        #   判断每一个世代的长度
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("epoch_step/epoch_step_val too few")

        #   构建数据集加载器。
        train_dataset   = FacenetDataset(input_shape, lines[:num_train], num_classes, random = True)
        val_dataset     = FacenetDataset(input_shape, lines[num_train:], num_classes, random = False)

        train_sampler   = None
        val_sampler     = None
        shuffle         = True
        
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            val_total_triple_loss,val_total_CE_loss,val_total_accuracy,LFW_Accuracy=fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda, LFW_loader, batch_size//3, lfw_eval_flag, fp16, scaler, save_period, save_dir, local_rank)
            
            # tensorboard 日志写入
            writer.add_scalar('val_total_triple_loss', val_total_triple_loss, epoch)
            writer.add_scalar('val_total_CE_loss', val_total_CE_loss, epoch)
            writer.add_scalar('val_total_accuracy', val_total_accuracy, epoch)
            writer.add_scalar('LFW_Accuracy', LFW_Accuracy, epoch)

        if local_rank == 0:
            loss_history.writer.close()
    writer.close()