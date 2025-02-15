import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from DrtNet_FA_Model import DrtNet_FA as kidney_seg
from DrtNet_FA_Model import CONFIGS as CONFIGS_seg
from trainer import trainer_synapse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./All_Data/organ_npz', help='root dir for data')
parser.add_argument('--list_dir', type=str, default='./All_Data/list', help='list dir')
parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-3, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--n_layer', type=int, default=4, help='the number of ResNet layers, default is four')
parser.add_argument('--vit_name', type=str, default='Config', help='select config')
parser.add_argument('--patch_size', type=int, default=32, help='vit_patches_size, default is 32')
args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")
    snapshot_path = os.path.join(model_dir, args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_layer' + str(args.n_layer)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.patch_size) if args.patch_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    config_vit = CONFIGS_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_layer = args.n_layer
    config_vit.patch_size = args.patch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = kidney_seg(config_vit, img_size=args.img_size).to(device)
    trainer = {'Synapse': trainer_synapse, }
    trainer[dataset_name](args, net, snapshot_path)
