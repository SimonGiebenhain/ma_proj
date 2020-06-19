import plotly.graph_objects as go
import numpy as np
import os.path as osp
from easydict import EasyDict
from psbody.mesh import Mesh



args = EasyDict()
args.dataset = 'CoMA'
args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, '..', 'spiralnet_plus', 'data', args.dataset)
template_fp = osp.join(args.data_fp, 'template', 'template.obj')
print(template_fp)

mesh = Mesh(filename=template_fp)

template_obj = go.Mesh3d(
                x=mesh.v[:, 0],
                y=mesh.v[:, 1],
                z=mesh.v[:, 2],
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

fig = go.Figure(data=template_obj)


fig.show()