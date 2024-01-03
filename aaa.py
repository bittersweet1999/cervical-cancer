from models.vit_moe import vit_base_patch16_224_in21k_multi_long_tail_MoE
import torch
model = vit_base_patch16_224_in21k_multi_long_tail_MoE(
                            embed_dim=512,
                            depth=4,
                            num_heads=2,
                            num_classes=2,
                            multi_tasks=6,long_tail=[True,False]*3,alpha=0.6,multi_gate=True).cuda()
a=torch.FloatTensor(4,256,512).cuda()
# model.eval()
out=model(a,task_id=None)
print(len(out),type(out),out.keys())
print(out[list(out.keys())[0]].size())
# print((out[0]==out[1]).sum()/out[1].numel())

# print(model.heads)
# print(model.heads)