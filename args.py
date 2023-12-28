import argparse
import os

def set_type(x,target_type):
    x = list(map(target_type,x.split(',')))
    return x  

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--lr-res', type=float, default=0.00005)
parser.add_argument('--lrf', type=float, default=0.1)

parser.add_argument('--lr-head', type=lambda x: set_type(x,float), default=0.00005)
parser.add_argument('--loss-weights',type=lambda x: set_type(x,float), default=1)

parser.add_argument('--weights', type=str, default='',
                    help='initial weights path')
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--where', default="./duofenlei", help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--base-path', default='/public_bme/data/jianght/datas/Pathology/class2')
parser.add_argument('--train-csv', default='train.csv')
parser.add_argument('--valid-csv', default='test.csv')
parser.add_argument('--positive-csv', default='test.csv')
parser.add_argument('--negative-csv', default='test.csv')
parser.add_argument('--head-idx', default=None, type=int)
parser.add_argument('--img-batch', default=100, type=int,help=' image numbers of a sample')

parser.add_argument('--arch', default='resnet34',help='name of resnet,such as resnet34 or resnet50')
parser.add_argument('--res-weights', default='')
parser.add_argument('--res-savedir', default='./resnet')

parser.add_argument('--multi-tasks',type=int,default=2,help='num of tasks')
parser.add_argument('--num-classes', type=lambda x: set_type(x,int), default=2 ,help='an integer or a list of integers')
parser.add_argument('--loss-fns',type= lambda x: set_type(x,str), default='CELoss' ,help='a str or a list of strs')
parser.add_argument('--tasks',type= lambda x: set_type(x,str), default='fungus,label' ,help='a str or a list of tasks')
parser.add_argument('--long-tails',type= lambda x: set_type(x,str), default='False,False' ,help='whether use long_tails or not')
parser.add_argument('--alpha',type=lambda x: set_type(x,float),default=1,help='the value of alpha if long-tail')

parser.add_argument('--cont', action='store_true',help='need to count high positive or not')
parser.add_argument('--show-tasks',type=lambda x: set_type(x,int),default=None,help='index of tasks to show the resluts')
parser.add_argument('--needpatch', action='store_true')
parser.add_argument('--backbone',default='vit',choices=['vit','TransMIL','vit_res','vit_moe'])
parser.add_argument('--reduction', default='mean',choices=['mean','sum','none'])
parser.add_argument('--logdir',required=True,help='dir to save error log')



parser.add_argument('--depth',type=int,default=12)
parser.add_argument('--gate-dim',type=int,default=None)
parser.add_argument('--moe_experts',type=int,default=16)
parser.add_argument("--local-rank","--local_rank", help="local device id on current node",type=int,default=None)
parser.add_argument('--use_weight', action='store_true',help='need to balance the loss or not')
parser.add_argument('--moe_top_k',type=lambda x: set_type(x,int),default=4,help='top k of per task')


def init_args(args):
    check_attrs = ['num_classes','loss_fns','tasks','long_tails','alpha','loss_weights','lr_head','moe_top_k']
    for attr in check_attrs:
        val = getattr(args,attr)
        assert isinstance(val, (int, float, str,list)) , f'expect type of {attr} in [int,float,str] ,but get {val} {type(val)}'

        if isinstance(val,list):
            if len(val) == 1:
                val *= args.multi_tasks
                setattr(args,attr,val)
            else:
                assert len(val) == args.multi_tasks, f'expect len of {attr} to be {args.multi_tasks} ,but get {len(val)}'
            continue
        else:
            val = [val] * args.multi_tasks
            setattr(args,attr,val)
            
    args.train_csv = os.path.join(args.base_path,args.train_csv)
    args.valid_csv = os.path.join(args.base_path,args.valid_csv)
    args.positive_csv = os.path.join(args.base_path,args.positive_csv)
    args.negative_csv = os.path.join(args.base_path,args.negative_csv)
    
    args.long_tails = [i == 'True' for i in args.long_tails] if isinstance(args.long_tails,list) else args.long_tails == 'True'
    args.task_id = [i for i in range(args.multi_tasks) if args.lr_head[i]>1e-8]
    print(args.task_id,'wdada')
    # parser.add_argument('--task-id',type= lambda x: set_type(x,int),default=None)
    
    if args.show_tasks is None:
        args.show_tasks = list(range(args.multi_tasks))
    
    return args

    
def get_args():
    args = parser.parse_args()
    init_args(args)

    return args

if __name__ == '__main__':
    args = get_args()
    print(args)

