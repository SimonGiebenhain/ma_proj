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


def get_geometric_object(v, f, color, color_min, color_max, offset, is_first=False):

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



def visualize_data(batch, reconstructions, std, mean, viz_width=9):
    # denormalize
    batch.x *= std
    batch.x += mean

    for i, rec in enumerate(reconstructions):
        reconstructions[i] = (rec * std) + mean

    rec_errors = [torch.norm(batch.x - rec, dim=2) for rec in reconstructions]
    min_error = 0
    max_errors = [torch.max(torch.max(rec_err)).item() for rec_err in rec_errors]
    max_error = max(max_errors)

    geometric_objs = []
    for i in range(viz_width):
        for j in range(len(reconstructions)+1):
            offset = [i*0.25, -j*0.35, 0]
            if j == 0:
                geometric_objs.append(get_geometric_object(v=np.squeeze(batch.x[i, :, :].numpy()),
                                            f=np.squeeze(batch.face[i, :, :].numpy()),
                                            color=np.zeros(batch.x.shape[0]),
                                            color_min=min_error,
                                            color_max=max_error,
                                            offset=offset,
                                            is_first=i == 0)
                           )
            else:
                v_pred = np.squeeze(reconstructions[j-1][i, :, :].detach().numpy())
                v_true = np.squeeze(batch.x[i, :, :].detach().numpy())
                rec_error = np.linalg.norm(v_pred-v_true, axis=1)
                geometric_objs.append(get_geometric_object(v=v_pred,
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

    fig = go.Figure(data=geometric_objs, layout=layout)
    fig.show()