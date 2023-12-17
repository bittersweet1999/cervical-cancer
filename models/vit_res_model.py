import torch
import torch.nn as nn
from models.vit_model_res18 import vit_base_patch16_224_in21k_multi_long_tail
from models.vit_moe import vit_multi_long_tail_MoE
from torchvision import models
from args import get_args

args = get_args()


class VitRes(nn.Module):
    def __init__(self,embed_dim=512,num_classes=args.num_classes, has_logits=False,multy_tasks=args.multi_tasks,head_idx=args.head_idx,long_tail=args.long_tails,alpha=args.alpha):
        super().__init__()
        resnet = models.__dict__[args.arch](pretrained=True)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.resnet = resnet
        embed_dim = list(dict(resnet.named_parameters()).values())[-1].shape[-1]
        print(f'embed_dim is {embed_dim}----------------------------------------------------------------')
        self.model = vit_base_patch16_224_in21k_multi_long_tail(embed_dim=embed_dim, num_classes=num_classes,
                                                                has_logits=has_logits, multy_tasks=multy_tasks,
                                                                head_idx=head_idx, long_tail=long_tail,
                                                                alpha=alpha)
        #self.embed_mean = torch.zeros(args.multi_tasks,embed_dim=512).numpy()

    def forward(self, x):
        B = x.shape[0]
        # print('sss',x.size())
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        features = self.resnet(x)

        # print(features.reshape(1,-1)[0,:30])
        # print(features[0,:10])
        features = torch.reshape(features, (B, int(features.shape[0] / B), features.shape[1]))
        preds = self.model(features)
        return preds


class VitResMoE(nn.Module):
    def __init__(self,embed_dim=512,num_classes=args.num_classes, has_logits=False,multi_tasks=args.multi_tasks,long_tail=args.long_tails,alpha=args.alpha,depth=args.depth,moe_gate_dim=args.gate_dim,num_experts_pertask=args.num_experts_pertask):
        super().__init__()
        resnet = models.__dict__[args.arch](pretrained=True)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.resnet = resnet
        embed_dim = list(dict(resnet.named_parameters()).values())[-1].shape[-1]
        print(f'embed_dim is {embed_dim}----------------------------------------------------------------')
        
        if args.gate_dim is None:
            gate_dim = embed_dim+2
        else:
            gate_dim = args.gate_dim
        
        self.model = vit_multi_long_tail_MoE(embed_dim=embed_dim, num_classes= num_classes, has_logits = has_logits, multi_tasks=multi_tasks,depth=depth,long_tail=long_tail,alpha=alpha,num_heads=16,\
                            moe_mlp_ratio=1,moe_experts=16,moe_top_k=4,moe_gate_dim=gate_dim,world_size=1,gate_return_decoupled_activation=False,
                            moe_gate_type="noisy_vmoe", vmoe_noisy_std=1, gate_task_specific_dim=-1,multi_gate=True,regu_experts_fromtask = False, 
                            num_experts_pertask = num_experts_pertask, num_tasks = -1, gate_input_ahead=False, regu_sem=False, sem_force=False, regu_subimage=False, 
                            expert_prune=False)
        #self.embed_mean = torch.zeros(args.multi_tasks,embed_dim=512).numpy()

    def forward(self, x,task_id=None):
        B = x.shape[0]
        # print('sss',x.size())
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        features = self.resnet(x)

        # print(features.reshape(1,-1)[0,:30])
        # print(features[0,:10])
        features = torch.reshape(features, (B, int(features.shape[0] / B), features.shape[1]))
        preds = self.model(features,task_id)
        return preds


if __name__ == '__main__':
    net = VitRes().to('cuda')
    n = torch.load("C:\\Users\\Administrator\\Desktop\\fsdownload\\resnet.pth")
    print(net.resnet.load_state_dict(n))
    print(net.model.load_state_dict(torch.load("C:\\Users\\Administrator\\Desktop\\fsdownload\\model-None-56.pth")['state_dict'],strict=False))
    print('1111')
