import plotly.graph_objects as go
from plotly.graph_objects import Layout
import numpy as np
import os.path as osp
from easydict import EasyDict
from psbody.mesh import Mesh
from utils import DataLoader
from datasets import MeshData



def construct_geometric_object(obj_fp, offset):
    mesh = Mesh(filename=obj_fp)

    template_obj = go.Mesh3d(
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
    return template_obj

def get_go_from_data(v, f, offset):

    template_obj = go.Mesh3d(
        x=v[:, 0] + offset[0],
        y=v[:, 1] + offset[1],
        z=v[:, 2] + offset[2],
        colorbar_title='z',
        colorscale=[[0, 'gold'],
                    [0.5, 'mediumturquoise'],
                    [1, 'magenta']],
        intensity=v[:, 2],
        i=f[0, :],
        j=f[1, :],
        k=f[2, :],
        showscale=True
    )
    return template_obj


args = EasyDict()
args.dataset = 'CoMA'
args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, '..', 'spiralnet_plus', 'data', args.dataset)
template_fp = osp.join(args.data_fp, 'template', 'template.obj')

meshdata = MeshData(args.data_fp,
                    template_fp,
                    split='interpolation',
                    test_exp='bareteeth')
test_loader = DataLoader(meshdata.test_dataset, batch_size=10, shuffle=True)
mean = meshdata.mean
std = meshdata.std
del meshdata

for i, data in enumerate(test_loader):
    batch = data
    if i == 1:
        break
del test_loader

# denormalize
batch.x *= std
batch.x += mean

print(batch.x.shape)

gos = []
idx = 0
for i in range(5):
    for j in range(2):
        offset = [i*0.25, j*0.35, 0]
        #gos.append(construct_geometric_object(template_fp, offset))
        gos.append(get_go_from_data(np.squeeze(batch.x[idx, :, :].numpy()),
                                    np.squeeze(batch.face[idx, :, :].numpy()),
                                    offset)
                   )
        idx += 1

layout = Layout(
    scene=dict(
        aspectmode='data'
    )
)

fig = go.Figure(data=gos, layout=layout)


fig.show()