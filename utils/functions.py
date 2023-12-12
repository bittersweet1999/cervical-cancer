import numpy as np
import torch
from torch import optim
import os

def calculate_loss(lossfn,pred,label,task_name,loss_weight,reduction,labels,cont):
    loss = lossfn(pred, label)
    error_name = []
    errors_num = 0       
    if task_name == 'label':
        if cont:
            pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]
            error = ((labels[-1].cpu() == 2) * (pred_classes.cpu() != label.cpu()))
            errors_num = error.sum().item()
            error_name = [labels[-2][i] + '\n' for i in range(error.size(0)) if error[i]]
            
    if reduction == 'none':
        if task_name == 'label':
            multilabel_msk = [ loss_weight if i == 2 else 1 if i == 1  else 0.5 for i in labels[-1] ] # -1: multi_labels -2:belong
        else :
            multilabel_msk = 1
        multilabel_weight = torch.Tensor(multilabel_msk).to(loss.device)
        loss = multilabel_weight * loss
        loss = loss.mean()

    return loss,errors_num,error_name

def get_optimizer(args,model):
    if args.backbone == 'vit':
        ignored_params = list(map(id, model.heads.parameters()))   # 返回的是parameters的 内存地址
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())     
        params_total = [{'params':model.heads[i].parameters(), 'lr': args.lr_head[i],'momentum': 0.9, 'weight_decay': 5e-6}for i in range(args.multi_tasks) if args.lr_head[i] > 1e-8 ]
        params_total.append({'params': base_params})
        print(len(params_total),params_total)
        optimizer = optim.SGD(params_total,args.lr, momentum=0.9, weight_decay=5e-6)
    elif args.backbone == 'TransMIL':
        optimizer = optim.SGD( model.parameters(), args.lr, momentum=0.9, weight_decay=5e-6)
    elif args.backbone == 'vit_res':
        ignored_params = list(map(id, model.model.heads.parameters()))   # 返回的是parameters的 内存地址
        print(ignored_params)
        base_params = filter(lambda p: id(p) not in ignored_params, model.model.parameters())
        res_params = model.resnet.parameters()
        params_total = [{'params':model.model.heads[i].parameters(), 'lr': args.lr_head[i],'momentum': 0.9, 'weight_decay': 5e-6}for i in range(args.multi_tasks) if args.lr_head[i] > 1e-8 ]
        params_total.append({'params': base_params})
        params_total.append({'params':res_params,'lr': args.lr_res,'momentum': 0.9, 'weight_decay': 5e-6})
        print(len(params_total),params_total)
        optimizer = optim.SGD( params_total, args.lr, momentum=0.9, weight_decay=5e-6)
    return optimizer

def load_state(args,model):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        if 'state_dict' in weights_dict.keys():
            print('with state_dict')
            if args.backbone == 'vit_res':
                #print(weights_dict['state_dict'].keys())
                len_clks = weights_dict['state_dict']['model.task_tokens'].shape[1]
                # del weights_dict['state_dict']['model.heads.0.weight']
                if args.multi_tasks != len_clks:
                    print(f'multi_tasks is {args.multi_tasks} but the cls is {len_clks}')
                    #fungus_state = torch.load('/public/home/jianght2023/pths_multi_longtail_fungus_0.7/model-None-59.pth')
                    for i in range(args.multi_tasks - len_clks):
                        weights_dict['state_dict']['model.task_tokens'] = torch.cat((weights_dict['state_dict']['model.task_tokens'],weights_dict['state_dict']['model.task_tokens'][:,-1].unsqueeze(0)),dim=1) 
                        #weights_dict['state_dict'][f'heads.{len_clks+i+1}.weight'] = fungus_state['state_dict']['heads.0.weight']
                        weights_dict['embed_mean'] = np.concatenate([model.model.embed_mean,weights_dict['embed_mean'][0].reshape(1,-1)])

                print(model.load_state_dict(weights_dict['state_dict'], strict=False))
                model.model.embed_mean = weights_dict['embed_mean']
                print(model.model.embed_mean.shape)
            else:
                model.embed_mean = weights_dict['embed_mean']
            #model.embed_mean = np.concatenate([model.embed_mean,model.embed_mean[0].reshape(1,-1)])
        else:
            print('without state_dict')
            print(model.load_state_dict(weights_dict, strict=False))
    return model