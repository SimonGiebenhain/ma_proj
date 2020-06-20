import pickle
import argparse
from copy import deepcopy
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torch_geometric.transforms as T
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewers

from reconstruction import AE, VAE, run, eval_error
from datasets import MeshData
from utils import utils, writer, DataLoader, mesh_sampling

parser = argparse.ArgumentParser(description='mesh variational autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--dataset', type=str, default='CoMA')
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[32, 32, 32, 64],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=16)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')
parser.add_argument('--lam', type=float, default=0.001)

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=300)

# others
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, '..', 'spiralnet_plus', 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, '..', 'spiralnet_plus', 'reconstruction', 'out', args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
if torch.cuda.is_available():
    device = torch.device('cuda', args.device_idx)
else:
    device = torch.device('cpu')
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

template_fp = osp.join(args.data_fp, 'template', 'template.obj')

# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform.pkl')
if not osp.exists(transform_fp):
    raise (Exception('transforms should already be present in ' + transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                            tmp['vertices'][idx],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]
del tmp['face']
del tmp['vertices']
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
del tmp['down_transform']
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]
del tmp


args.batch_size = 100
# generate random measurement matrix
template_mesh = Mesh(filename=template_fp)
meshdata = MeshData(args.data_fp,
                    template_fp,
                    split=args.split,
                    test_exp=args.test_exp)
test_loader = DataLoader(meshdata.test_dataset, batch_size=args.batch_size)

measurement_sizes = [25, 50, 100, 200, 500, 1000, 2000]
error_matrix = np.zeros([len(measurement_sizes), 4])
for itr, msize in enumerate(measurement_sizes):

    model = VAE(args.in_channels, args.out_channels, args.latent_channels,
                spiral_indices_list, down_transform_list,
                up_transform_list, lam=args.lam).to(device)
    del up_transform_list, down_transform_list, spiral_indices_list
    #print('Number of parameters: {}'.format(utils.count_parameters(model)))
    #print(model)

    # load pretrained mesh VAE model
    # TODO read model_checkpoint path from command line
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(osp.join(args.checkpoints_dir, 'vae_checkpoint.pt'), map_location=torch.device('cpu'))['model_state_dict'])
    else:
        model.load_state_dict(torch.load(osp.join(args.checkpoints_dir, 'vae_checkpoint.pt'))['model_state_dict'])

    model.eval()


    # get a random test example
    for i,data in enumerate(test_loader):
        batch = torch.squeeze(data.x[:, :, :])
        if i == itr:#42#61
            break

    nv = template_mesh.v.shape[0]
    # sample simple measurement matrix, which just subselects a set of vertices,
    # this enables to use the a regular norm for the gradient descent
    A = torch.zeros([args.batch_size, msize, nv])
    for i in range(args.batch_size):
        B = np.random.choice(nv, size=[msize], replace=False)
        for j in range(msize):
            A[i, j, B[j]] = 1

    A.requires_grad_(False)

    # get measurements
    measurements = torch.matmul(A, batch)
    measurements.requires_grad_(False)

    # set up optimizer w.r.t. latent representation
    latent_rep = torch.zeros([measurements.shape[0], args.latent_channels])
    latent_rep.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_rep],
                                 lr=0.1)

    #### Different Methods of reconstruction ####
    # 1) optimize w.r.t. latent representation
    for i in range(500):
        optimizer.zero_grad()
        out = model.decode(latent_rep)
        loss = F.l1_loss(torch.matmul(A, out), measurements, reduction='mean')
        #print(loss.item())
        loss.backward()
        optimizer.step()

    # Denormalize
    model.eval()

    errors = []
    out *= meshdata.std
    out += meshdata.mean
    batch *= meshdata.std
    batch += meshdata.mean
    out *= 1000
    batch *= 1000

    tmp_error = torch.sqrt(
        torch.sum((out - batch) ** 2,
                  dim=2))  # [num_graphs, num_nodes]

    mean_error = tmp_error.view((-1,)).mean()
    std_error = tmp_error.view((-1,)).std()
    median_error = tmp_error.view((-1,)).median()
    print('Number of measurements {}'.format(msize))
    message = 'Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error,
                                                     median_error)
    print(message)
    error_matrix[itr, 0] = msize
    error_matrix[itr, 1] = mean_error
    error_matrix[itr, 2] = std_error
    error_matrix[itr, 3] = median_error

np.save(osp.join(args.out_dir, 'compressed_sensing_err'), error_matrix)
