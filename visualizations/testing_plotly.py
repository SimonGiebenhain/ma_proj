import plotly.graph_objects as go
from plotly.graph_objects import Layout
import numpy as np
import os.path as osp
from easydict import EasyDict
from psbody.mesh import Mesh
from utils import DataLoader
from datasets import MeshData
import torch
from reconstruction import AE, AD, VAE
import pickle
import utils
import plotly.express as px
from compressed_sensing_utils import gen_random_A, gen_binary_A, optimize_latent_rep, eval_reconstruction


from visualizations_utils import visualize_data


def construct_geometric_object(obj_fp, offset):
    mesh = Mesh(filename=obj_fp)

    obj = go.Mesh3d(
        x=mesh.v[:, 0] + offset[0],
        y=mesh.v[:, 1] + offset[1],
        z=mesh.v[:, 2] + offset[2],
        colorbar_title='z',
        colorscale=[[0, 'gold'],
                    [0.5, 'mediumturquoise'],
                    [1, 'magenta']],
        intensity=mesh.v[:, 2],
        i=mesh.f[:, 0],
        j=mesh.f[:, 1],
        k=mesh.f[:, 2],
        showscale=True
    )
    return obj

#[[0, "rgb(0,60,170)"],
#                    [0.25, "rgb(5,255,255)"],
#                    [0.5, "rgb(255,255,0)"],
#                    [0.75, "rgb(250,0,0)"],
#                    [1, "rgb(128,0,0)"]],
def get_go_from_data(v, f, color, color_min, color_max, offset, is_first=False):

    template_obj = go.Mesh3d(
        x=v[:, 0] + offset[0],
        y=v[:, 1] + offset[1],
        z=v[:, 2] + offset[2],
        colorbar_title='z',
        colorscale=[[0, "rgb(255,255,255)"],
                    [0.3, "rgb(255,210,0)"],
                    [0.6, "rgb(200,0,0)"],
                    [1, "rgb(100,0,0)"]],
        intensity=color,
        cmin=color_min,
        cmax=color_max,
        i=f[0, :],
        j=f[1, :],
        k=f[2, :],
        showscale=is_first,
        lighting=dict(ambient=0.45, diffuse=0.5, specular=0.5, roughness=0.4),
        lightposition=dict(x=2, y=2, z=2)
    )
    return template_obj


args = EasyDict()
args.dataset = 'CoMA'
args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, '..', 'spiralnet_plus', 'data', args.dataset)
template_fp = osp.join(args.data_fp, 'template', 'template.obj')
args.out_dir = osp.join(args.work_dir, '..', 'spiralnet_plus', 'reconstruction', 'out', 'interpolation_exp')
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')

args.out_channels = [32, 32, 32, 64]
args.latent_channels = 16
args.in_channels = 3
args.seq_length = [9, 9, 9, 9]
args.dilation = [1, 1, 1, 1]


if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')

transform_fp = osp.join(args.data_fp, 'transform.pkl')

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


meshdata = MeshData(args.data_fp,
                    template_fp,
                    split='interpolation',
                    test_exp='bareteeth')

test_loader = DataLoader(meshdata.test_dataset, batch_size=100, shuffle=True)
mean = meshdata.mean
std = meshdata.std
del meshdata

for i, (data, idx) in enumerate(test_loader):
    batch = data
    if i == 0:
        break
del test_loader

#### AE ####
model_ae = AE(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list).to(device)

model_ae.load_state_dict(torch.load(osp.join(args.checkpoints_dir, 'ae_checkpoint_300.pt'),
                                    map_location=torch.device('cpu')
                                    )['model_state_dict'])
model_ae.eval()
rec_ae, _ = model_ae(batch.x)
del model_ae


#### regularized AE ####
model_reg_ae = AE(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list, lam=0.001).to(device)

model_reg_ae.load_state_dict(torch.load(osp.join(args.checkpoints_dir, 'ae_reg_checkpoint.pt'),
                                    map_location=torch.device('cpu')
                                    )['model_state_dict'])
model_reg_ae.eval()
rec_reg_ae, _ = model_reg_ae(batch.x)
del model_reg_ae

#### AD ####
model_ad = AD(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list[-1].size(0),
           up_transform_list, lam=0.001).to(device)


measurements = batch.x
measurements.requires_grad_(False)


rec_ad = optimize_latent_rep(model_ad, None, measurements, args.latent_channels, device)

#### VAE ####
#model_vae = VAE(args.in_channels, args.out_channels, args.latent_channels,
#           spiral_indices_list, down_transform_list,
#           up_transform_list, lam=0.001).to(device)
#
#model_vae.load_state_dict(torch.load(osp.join(args.checkpoints_dir, 'vae_checkpoint.pt'),
#                                     map_location=torch.device('cpu')
#                                     )['model_state_dict'])
#model_vae.eval()
#_, rec_vae, _, _ = model_vae(batch.x, also_give_map=True)
#del model_vae

visualize_data(batch, [rec_ae, rec_reg_ae, rec_ad], std, mean, viz_width=9)