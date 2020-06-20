import plotly.graph_objects as go
from plotly.graph_objects import Layout
import numpy as np
import os.path as osp
from easydict import EasyDict
from psbody.mesh import Mesh
from utils import DataLoader
from datasets import MeshData
import torch
from reconstruction import AE, VAE
import pickle
import utils
import plotly.express as px


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

for i, data in enumerate(test_loader):
    batch = data
    if i == 1:
        break
del test_loader

# load AE
model_ae = AE(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list).to(device)

model_ae.load_state_dict(torch.load(osp.join(args.checkpoints_dir, 'ae_checkpoint_300.pt'),
                                    map_location=torch.device('cpu')
                                    )['model_state_dict'])
model_ae.eval()

#load  VAE
model_vae = VAE(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list, lam=0.001).to(device)

model_vae.load_state_dict(torch.load(osp.join(args.checkpoints_dir, 'vae_checkpoint.pt'),
                                     map_location=torch.device('cpu')
                                     )['model_state_dict'])
model_vae.eval()

#reconstruct
rec_ae = model_ae(batch.x)
_, rec_vae, _, _ = model_vae(batch.x)

# denormalize
batch.x *= std
batch.x += mean

rec_ae *= std
rec_ae += mean

rec_vae *= std
rec_vae += mean

rec_errors = torch.norm(batch.x - rec_ae, dim=2)
min_error = 0
max_error = torch.max(torch.max(rec_errors)).item()

gos = []
for i in range(9):
    for j in range(3):
        offset = [i*0.25, -j*0.35, 0]
        if j == 0:
            gos.append(get_go_from_data(v=np.squeeze(batch.x[i, :, :].numpy()),
                                        f=np.squeeze(batch.face[i, :, :].numpy()),
                                        color=np.zeros(batch.x.shape[0]),
                                        color_min=min_error,
                                        color_max=max_error,
                                        offset=offset)
                       )
        if j == 1:
            v_pred = np.squeeze(rec_ae[i, :, :].detach().numpy())
            v_true = np.squeeze(batch.x[i, :, :].detach().numpy())
            rec_error = np.linalg.norm(v_pred-v_true, axis=1)
            gos.append(get_go_from_data(v=v_pred,
                                        f=np.squeeze(batch.face[i, :, :].numpy()),
                                        color=rec_error,
                                        color_min=min_error,
                                        color_max=max_error,
                                        offset=offset,
                                        is_first=i==0)
                       )
        if j == 2:
            v_pred = np.squeeze(rec_vae[i, :, :].detach().numpy())
            v_true = np.squeeze(batch.x[i, :, :].detach().numpy())
            rec_error = np.linalg.norm(v_pred - v_true, axis=1)
            gos.append(get_go_from_data(v=v_pred,
                                        f=np.squeeze(batch.face[i, :, :].numpy()),
                                        color=rec_error,
                                        color_min=min_error,
                                        color_max=max_error,
                                        offset=offset)
                       )


layout = Layout(
    autosize=True,
    scene=dict(
        aspectmode='data'
    ),
    margin=dict(
        l=50,
        r=50,
        b=0,
        t=0,
        pad=4
    ),
    scene_camera=dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=2)
    ),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig = go.Figure(data=gos, layout=layout)


fig.show()