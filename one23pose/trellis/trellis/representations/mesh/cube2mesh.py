# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicube import FlexiCubes

import torch
import trimesh
import numpy as np

# Dependency for mesh cleaning and hole fix
# from ...utils.random_utils import sphere_hammersley_sequence
# import utils3d
# import igraph
# from tqdm import tqdm
# from pymeshfix import _meshfix

class MeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.vertex_normal = self.comput_v_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None
        
    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        return face_normals[:, None, :].repeat(1, 3, 1)
                
    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals

    def to_trimesh(self, transform_pose=False):
        """
        Convert the mesh to a trimesh.Trimesh object.
        Args:
            transform_pose (bool): If True, transform the vertices to change coordinate system
        Returns:
            trimesh.Trimesh: The converted mesh
        """
        # Convert vertices and faces to numpy arrays
        vertices = self.vertices.detach().cpu().numpy()
        faces = self.faces.detach().cpu().numpy()
        
        # Apply coordinate transformation if requested
        if transform_pose:
            transform_matrix = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
            vertices = vertices @ transform_matrix
            
            # Also transform the normals if they exist
            vertex_normals = self.vertex_normal.detach().cpu().numpy() @ transform_matrix
        else:
            vertex_normals = self.vertex_normal.detach().cpu().numpy()
        
        # Prepare vertex colors if they exist
        vertex_colors = None
        if self.vertex_attrs is not None:
            vertex_colors = self.vertex_attrs[:, :3].detach().cpu().numpy()
            # Ensure colors are in [0, 255] range
            if vertex_colors.max() <= 1.0:
                vertex_colors = (vertex_colors * 255).astype(np.uint8)
        
        # Create the trimesh mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors
        )
        
        return mesh

    @staticmethod
    def from_trimesh(mesh, device='cuda'):
        # Convert scene to mesh if necessary
        if hasattr(mesh, 'geometry'):
            # If it's a scene, get the first mesh
            # Assuming the scene has at least one mesh
            mesh_name = list(mesh.geometry.keys())[0]
            mesh = mesh.geometry[mesh_name]
    
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.int64)
        
        vertex_attrs = None
        if mesh.visual.vertex_colors is not None:
            vertex_attrs = torch.tensor(mesh.visual.vertex_colors, dtype=torch.float32) / 255.0
            print(vertex_attrs)
            vertex_attrs = vertex_attrs[:, :3]
        else:
            vertex_attrs = torch.zeros((vertices.shape[0], 3), dtype=torch.float32)
        return MeshExtractResult(vertices, faces, vertex_attrs)
    
    def to(self, device):
        self.vertices = self.vertices.to(device)
        self.faces = self.faces.to(device)
        if self.vertex_attrs is not None:
            self.vertex_attrs = self.vertex_attrs.to(device)
        self.face_normal = self.face_normal.to(device)
        self.vertex_normal = self.vertex_normal.to(device)
        return self

    def subdivide(self):
        """
        Subdivide the mesh by splitting each triangle into four smaller triangles.
        """
        new_vertices = []
        new_faces = []
        vertex_map = {}

        def get_midpoint(v1, v2):
            edge = tuple(sorted((v1, v2)))
            if edge not in vertex_map:
                midpoint = (self.vertices[v1] + self.vertices[v2]) / 2
                vertex_map[edge] = len(new_vertices)
                new_vertices.append(midpoint)
            return vertex_map[edge]

        for face in self.faces:
            v0, v1, v2 = face
            a = get_midpoint(v0.item(), v1.item())
            b = get_midpoint(v1.item(), v2.item())
            c = get_midpoint(v2.item(), v0.item())

            new_faces.append([v0.item(), a, c])
            new_faces.append([v1.item(), b, a])
            new_faces.append([v2.item(), c, b])
            new_faces.append([a, b, c])

        new_vertices = torch.stack(new_vertices)
        new_faces = torch.tensor(new_faces, dtype=torch.long)

        self.vertices = torch.cat([self.vertices, new_vertices], dim=0)
        self.faces = new_faces
        self.face_normal = self.comput_face_normals(self.vertices, self.faces)
        self.vertex_normal = self.comput_v_normals(self.vertices, self.faces)
        self.vertex_attrs = torch.zeros((self.vertices.shape[0], 3), dtype=torch.float32)
        
class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True):
        '''
        a model to generate a mesh from sparse features structures using flexicube
        '''
        super().__init__()
        self.device=device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self._calc_layout()
    
    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            '''
            6 channel color including normal map
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start
        
    def get_layout(self, feats : torch.Tensor, name : str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats : SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        # add sdf bias to verts_attrs
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
            
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training)
        
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:,:20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
        return mesh
