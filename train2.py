import os
import math
import argparse
from utils.functions import get_optimizer,load_state

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from args import get_args
from my_dataset import MultiDataSet,MultiDataSetMoE
from models.vit_model_res18 import vit_base_patch16_224_in21k_multi_long_tail as create_model
from models.vit_res_model import VitRes,VitResMoE
from utils.utils import  train_one_epoch_multi_longtail_weight,evaluate_multi_longtail_weight,train_one_epoch_multi_moe,evaluate_multi_moe
# from models.TransMIL import TransMIL

import pandas as pd
import random
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def seed_reproducer(seed=2022):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
def main(args):
    seed_reproducer(9)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.where) is False:
        print(f'making dir {args.where}')
        os.makedirs(args.where)

    tb_writer = SummaryWriter(args.where)

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([#transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        "val": transforms.Compose([#transforms.Resize(256),
                                   #transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])}


        # 实例化训练数据集
    train_dataset = MultiDataSet(data=pd.read_csv(args.valid_csv),
                              #images_class=train_images_label,
                              transforms=data_transform['train'],
                              img_batch=args.img_batch,
                              head_idx=args.head_idx,
                              tasks=args.tasks,
                              need_patch=args.needpatch
                                 )
    #print(train_dataset)
    # 实例化验证数据集
    val_dataset = MultiDataSet(data=pd.read_csv(args.valid_csv),
                                #images_class=val_images_label,
                                transforms=data_transform['val'],
                                img_batch=args.img_batch,
                                head_idx=args.head_idx,
                                tasks=args.tasks,
                                need_patch=args.needpatch)
    
    # positive_dataset = MultiDataSet(data=pd.read_csv(args.positive_csv),
    #                           #images_class=train_images_label,
    #                           transforms=data_transform['train'],
    #                           img_batch=args.img_batch,
    #                           head_idx=args.head_idx,
    #                           tasks=args.tasks,
    #                           need_patch=args.needpatch
    #                              )

    print(len(train_dataset),len(val_dataset))
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # nw = 0
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,)
                                               #collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,)
                                             #collate_fn=val_dataset.collate_fn)

    # positive_loader = torch.utils.data.DataLoader(positive_dataset,
    #                                         batch_size=batch_size,
    #                                         shuffle=True,
    #                                         pin_memory=True,
    #                                         num_workers=nw,)
    if args.backbone == 'vit':
        model = create_model(embed_dim=512,num_classes=args.num_classes, has_logits=False,multy_tasks=args.multi_tasks,head_idx=args.head_idx,long_tail=args.long_tails,alpha=args.alpha).to(device)
    elif args.backbone == 'TransMIL':
        model = TransMIL(n_classes=args.num_classes).to(device)
    elif args.backbone == 'vit_res':
        model = VitRes(embed_dim=512,num_classes=args.num_classes, has_logits=False,multy_tasks=args.multi_tasks,head_idx=args.head_idx,long_tail=args.long_tails,alpha=args.alpha).to(device)
    elif args.backbone == 'vit_moe':
        model = VitResMoE(embed_dim=512,num_classes=args.num_classes, has_logits=False,multi_tasks=args.multi_tasks,long_tail=args.long_tails,alpha=args.alpha).to(device)
        print('dsawww')

    model = load_state(args,model)
    
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

   # model.heads[1-args.head_idx].requires_grad_(False)


    optimizer = get_optimizer(args,model)


    #optimizer = optim.SGD( model.heads[args.head_idx].parameters(), args.lr_head, momentum=0.9, weight_decay=5e-6)
    #
    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    istrain_list = [True  if args.lr_head[i] > 1e-8 else False for i in range(args.multi_tasks)]
    print(istrain_list)


    if args.cont:
        with open(args.logdir,'w') as f:
            f.write('')
    train_error_dict = {}
    pos_error_dict = {}
    train_error_list = []

    for epoch in range(args.epochs):

        # train
#
    #     if args.backbone == 'vit_res':
    #         train_loss, train_accs, train_error_list = train_one_epoch_multi_longtail_weight(model=model,
    #                                                 optimizer=optimizer,
    #                                                 data_loader=train_loader,
    #                                                 device=device,
    #                                                 epoch=epoch,
    #                                                 multi_tasks=args.multi_tasks,
    #                                                 istrain = istrain_list,
    #                                                 cont=args.cont
    #                                                 )
    #     elif args.backbone == 'vit_moe':
    #             train_loss, train_accs, train_error_list = train_one_epoch_multi_moe(model=model,
    #                                         optimizer=optimizer,
    #                                         data_loader=train_loader,
    #                                         device=device,
    #                                         epoch=epoch,
    #                                         multi_tasks=args.multi_tasks,
    #                                         istrain = istrain_list,
    #                                         cont=args.cont
    #                                         )
    # #   
    #     scheduler.step()

    #     if args.cont:
    #         train_loss, train_accs, train_error_list = train_res
    #     else:
    #         train_loss, train_accs = train_res 

        # validate
        # val_loss, val_accs,aucs, F1s, accuracys, sensitivitys, specificitys = evaluate_auc(model=model,
        #                              data_loader=val_loader,
        #                              device=device,
        #                              epoch=epoch,
        #                              multi_tasks=args.multi_tasks,
        #                              name='valid'
        #                             )
        # print('val_acc',val_loss, val_accs,aucs, F1s, accuracys, sensitivitys, specificitys)
##            
        if args.backbone == 'vit_res':
            val_loss, val_accs, _ = evaluate_multi_longtail_weight(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch,
                            multi_tasks=args.multi_tasks,
                            name='val',
                            cont=True
                        )
        elif args.backbone == 'vit_moe':
            val_loss, val_accs, _ = evaluate_multi_moe(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch,
                            multi_tasks=args.multi_tasks,
                            name='val',
                            cont=True
                        )
        print('val_acc',val_loss, val_accs)
        
        

        # pos_res = evaluate_multi_longtail_weight(model=model,
        #                             data_loader=positive_loader,
        #                             device=device,
        #                             epoch=epoch,
        #                             multi_tasks=args.multi_tasks,
        #                             name='pos',
        #                             cont=args.cont
        #                         )
        # if args.cont:
        #     pos_loss, pos_accs, pos_error_list = pos_res
        # else:
        #     pos_loss, pos_accs = pos_res
        # print('pos_acc',pos_loss, pos_accs )
    


        for i in range(args.multi_tasks):
            # ...
                tb_writer.add_scalar(f'train_loss_{i}',train_loss[i],epoch)
                tb_writer.add_scalar(f'train_acc_{i}',train_accs[i],epoch)
                tb_writer.add_scalar(f'val_loss_{i}',val_loss[i],epoch)
                tb_writer.add_scalar(f'val_acc_{i}',val_accs[i],epoch)
                # tb_writer.add_scalar(f'pos_loss_{i}',pos_loss[i],epoch)
                # tb_writer.add_scalar(f'pos_acc_{i}',pos_accs[i],epoch)

        # if args.cont:
        #     tb_writer.add_scalar(f'train_error_highpos',len(train_error_list),epoch)
        #     tb_writer.add_scalar(f'val_error_highpos',len(pos_error_list),epoch)

        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

        if True in args.long_tails:
            #print('saving with mean')
            state_dicts={
                'state_dict':model.state_dict(),
                'embed_mean':model.model.embed_mean if args.backbone in ('vit_res','vit_moe') else model.embed_mean
            }
            #print(model.embed_mean)
            torch.save(state_dicts,args.where+"/model-{}-{}.pth".format(args.head_idx,epoch))
        else:torch.save(model.state_dict(), args.where+"/model-{}-{}.pth".format(args.head_idx,epoch))


if __name__ == '__main__':

    opt = get_args()
    print(opt)

    main(opt)
