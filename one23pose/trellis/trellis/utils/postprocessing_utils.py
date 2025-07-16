from typing import *
import numpy as np
import trimesh
import trimesh.visual
import xatlas
from PIL import Image
import cv2
import igraph
import pyvista as pv

from pymeshfix import _meshfix

from .render_utils import render_condition_images, render_multiview
from ..renderers import GaussianRenderer
from ..representations import MeshExtractResult, Gaussian

from .random_utils import sphere_hammersley_sequence

import torch
import utils3d
from tqdm import tqdm
import nvdiffrast.torch as dr
import torchvision
from ..representations import MeshExtractResult

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

@torch.no_grad()
def _fill_holes(
    verts,
    faces,
    max_hole_size=0.04,
    max_hole_nbe=32,
    resolution=128,
    num_views=500,
    debug=False,
    verbose=False
):
    """
    Rasterize a mesh from multiple views and remove invisible faces.
    Also includes postprocessing to:
        1. Remove connected components that are have low visibility.
        2. Mincut to remove faces at the inner side of the mesh connected to the outer side with a small hole.

    Args:
        verts (torch.Tensor): Vertices of the mesh. Shape (V, 3).
        faces (torch.Tensor): Faces of the mesh. Shape (F, 3).
        max_hole_size (float): Maximum area of a hole to fill.
        resolution (int): Resolution of the rasterization.
        num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """
    # Construct cameras
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws = torch.tensor(yaws).cuda()
    pitchs = torch.tensor(pitchs).cuda()
    radius = 2.0
    fov = torch.deg2rad(torch.tensor(40)).cuda()
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    views = []
    for (yaw, pitch) in zip(yaws, pitchs):
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda().float() * radius
        view = utils3d.torch.view_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        views.append(view)
    views = torch.stack(views, dim=0)

    # Rasterize
    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    rastctx = utils3d.torch.RastContext(backend='cuda')#gengzheng : segmentation fault happened
    for i in tqdm(range(views.shape[0]), total=views.shape[0], disable=not verbose, desc='Rasterizing'):
        view = views[i]
        buffers = utils3d.torch.rasterize_triangle_faces(
            rastctx, verts[None], faces, resolution, resolution, view=view, projection=projection
        )
        face_id = buffers['face_id'][0][buffers['mask'][0] > 0.95] - 1
        face_id = torch.unique(face_id).long()
        visblity[face_id] += 1
    visblity = visblity.float() / num_views
    
    # Mincut
    ## construct outer faces
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    connected_components = utils3d.torch.compute_connected_components(faces, edges, face2edge)
    outer_face_indices = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    for i in range(len(connected_components)):
        outer_face_indices[connected_components[i]] = visblity[connected_components[i]] > min(max(visblity[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)
    
    ## construct inner faces
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if verbose:
        tqdm.write(f'Found {inner_face_indices.shape[0]} invisible faces')
    if inner_face_indices.shape[0] == 0:
        return verts, faces
    
    ## Construct dual graph (faces as nodes, edges as edges)
    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_edges_weights = torch.norm(verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1)
    if verbose:
        tqdm.write(f'Dual graph: {dual_edges.shape[0]} edges')

    ## solve mincut problem
    ### construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])
    g.add_edges(dual_edges.cpu().numpy())
    g.es['weight'] = dual_edges_weights.cpu().numpy()
    
    ### source and target
    g.add_vertex('s')
    g.add_vertex('t')
    
    ### connect invisible faces to source
    g.add_edges([(f, 's') for f in inner_face_indices], attributes={'weight': torch.ones(inner_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})
    
    ### connect outer faces to target
    g.add_edges([(f, 't') for f in outer_face_indices], attributes={'weight': torch.ones(outer_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})
                
    ### solve mincut
    cut = g.mincut('s', 't', (np.array(g.es['weight']) * 1000).tolist())
    remove_face_indices = torch.tensor([v for v in cut.partition[0] if v < faces.shape[0]], dtype=torch.long, device=faces.device)
    if verbose:
        tqdm.write(f'Mincut solved, start checking the cut')
    
    ### check if the cut is valid with each connected component
    to_remove_cc = utils3d.torch.compute_connected_components(faces[remove_face_indices])
    if debug:
        tqdm.write(f'Number of connected components of the cut: {len(to_remove_cc)}')
    valid_remove_cc = []
    cutting_edges = []
    for cc in to_remove_cc:
        #### check if the connected component has low visibility
        visblity_median = visblity[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f'visblity_median: {visblity_median}')
        if visblity_median > 0.25:
            continue
        
        #### check if the cuting loop is small enough
        cc_edge_indices, cc_edges_degree = torch.unique(face2edge[remove_face_indices[cc]], return_counts=True)
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)]
        if len(cc_new_boundary_edge_indices) > 0:
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(edges[cc_new_boundary_edge_indices])
            cc_new_boundary_edges_cc_center = [verts[edges[cc_new_boundary_edge_indices[edge_cc]]].mean(dim=1).mean(dim=0) for edge_cc in cc_new_boundary_edge_cc]
            cc_new_boundary_edges_cc_area = []
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]] - cc_new_boundary_edges_cc_center[i]
                _e2 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]] - cc_new_boundary_edges_cc_center[i]
                cc_new_boundary_edges_cc_area.append(torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5)
            if debug:
                cutting_edges.append(cc_new_boundary_edge_indices)
                tqdm.write(f'Area of the cutting loop: {cc_new_boundary_edges_cc_area}')
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue
            
        valid_remove_cc.append(cc)
        
    if debug:
        face_v = verts[faces].mean(dim=1).cpu().numpy()
        vis_dual_edges = dual_edges.cpu().numpy()
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]
        if len(valid_remove_cc) > 0:
            vis_colors[remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()] = [255, 0, 0]
        utils3d.io.write_ply('dbg_dual.ply', face_v, edges=vis_dual_edges, vertex_colors=vis_colors)
        
        vis_verts = verts.cpu().numpy()
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
        utils3d.io.write_ply('dbg_cut.ply', vis_verts, edges=vis_edges)
        
    
    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
        if verbose:
            tqdm.write(f'Removed {(~mask).sum()} faces by mincut')
    else:
        if verbose:
            tqdm.write(f'Removed 0 faces by mincut')
            
    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
    verts, faces = torch.tensor(verts, device='cuda', dtype=torch.float32), torch.tensor(faces, device='cuda', dtype=torch.int32)

    return verts, faces

SR_cache = None


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def run_sr_fast(source_pils, scale=4):
    from PIL import Image
    import numpy as np
    global SR_cache
    from aura_sr import AuraSR

    target_size = (512, 512)

    if SR_cache is not None:
        upsampler = SR_cache
    else:
        upsampler = AuraSR.from_pretrained("fal/AuraSR-v2")
    
    ret_pils = []
    for idx, img_pils in enumerate(tqdm(source_pils, desc='Image Upsampling')):
        np_in = isinstance(img_pils, np.ndarray)
        assert isinstance(img_pils, (Image.Image, np.ndarray))
        
        if np_in:
            img_pil = Image.fromarray(img_pils)
        else:
            img_pil = img_pils
        
        img_resized = img_pil.resize(target_size, Image.Resampling.LANCZOS)
        output = upsampler.upscale_4x_overlapped(img_resized)
        
        ret_pils.append(np.array(output) if np_in else output)
    
    if SR_cache is None:
        SR_cache = upsampler
    
    return ret_pils

def normalize_trimesh(mesh):
    # Calculate the mesh centroid and bounding box extents
    centroid = mesh.centroid
    # Determine the scale based on the largest extent to fit into unit cube
    # Normalizing: Center and scale the vertices
    mesh.vertices -= centroid

    extents = mesh.extents
    scale = max(extents)
    mesh.vertices /= scale

    return mesh

def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal['fast', 'opt'] = 'opt',
    lambda_tv: float = 1e-2,
    verbose: bool = False,
):
    """
    Bake texture to a mesh from multiple observations.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
        texture_size (int): Size of the texture.
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        mode (Literal['fast', 'opt']): Mode of texture baking.
        lambda_tv (float): Weight of total variation loss in optimization.
        verbose (bool): Whether to print progress.
    """
    vertices = torch.tensor(vertices).cuda()
    faces = torch.tensor(faces.astype(np.int32)).cuda()
    uvs = torch.tensor(uvs).cuda()
    observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
    masks = [torch.tensor(m>0).bool().cuda() for m in masks]
    views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
    projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).cuda(), near, far) for intr in intrinsics]

    if mode == 'fast':
        texture = torch.zeros((texture_size * texture_size, 3), dtype=torch.float32).cuda()
        texture_weights = torch.zeros((texture_size * texture_size), dtype=torch.float32).cuda()
        rastctx = utils3d.torch.RastContext(backend='cuda')
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(observations), disable=not verbose, desc='Texture baking (fast)'):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                uv_map = rast['uv'][0].detach().flip(0)
                mask = rast['mask'][0].detach().bool() & masks[0]
            
            # nearest neighbor interpolation
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
            texture_weights = texture_weights.scatter_add(0, idx, torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device))

        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # inpaint
        mask = (texture_weights == 0).cpu().numpy().astype(np.uint8).reshape(texture_size, texture_size)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    elif mode == 'opt':
        rastctx = utils3d.torch.RastContext(backend='cuda')
        observations = [observations.flip(0) for observations in observations]
        masks = [m.flip(0) for m in masks]
        _uv = []
        _uv_dr = []
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): UV'):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                _uv.append(rast['uv'].detach())
                _uv_dr.append(rast['uv_dr'].detach())

        texture = torch.nn.Parameter(torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).cuda())
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return start_lr * (end_lr / start_lr) ** (step / total_steps)

        def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
        
        def tv_loss(texture):
            return torch.nn.functional.l1_loss(texture[:, :-1, :, :], texture[:, 1:, :, :]) + \
                   torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])
    
        total_steps = 2500
        with tqdm(total=total_steps, disable=not verbose, desc='Texture baking (opt): optimizing') as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                selected = np.random.randint(0, len(views))
                uv, uv_dr, observation, mask = _uv[selected], _uv_dr[selected], observations[selected], masks[selected]
                render = dr.texture(texture, uv, uv_dr)[0]
                loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture)
                loss.backward()
                optimizer.step()
                # annealing
                optimizer.param_groups[0]['lr'] = cosine_anealing(optimizer, step, total_steps, 1e-2, 1e-5)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
        texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        mask = 1 - utils3d.torch.rasterize_triangle_faces(
            rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
        )['mask'][0].detach().cpu().numpy().astype(np.uint8)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    return texture

def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    fill_holes: bool = True,
    fill_holes_max_hole_size: float = 0.04,
    fill_holes_max_hole_nbe: int = 32,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    debug: bool = False,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_hole_size (float): Maximum area of a hole to fill.
        fill_holes_max_hole_nbe (int): Maximum number of boundary edges of a hole to fill.
        fill_holes_resolution (int): Resolution of the rasterization.
        fill_holes_num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(f'Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Simplify
    if simplify and simplify_ratio > 0:
        mesh = pv.PolyData(vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1))
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose:
            tqdm.write(f'After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Remove invisible faces
    if fill_holes:
        vertices, faces = torch.tensor(vertices).cuda(), torch.tensor(faces.astype(np.int32)).cuda()
        vertices, faces = _fill_holes(
            vertices, faces,
            max_hole_size=fill_holes_max_hole_size,
            max_hole_nbe=fill_holes_max_hole_nbe,
            resolution=fill_holes_resolution,
            num_views=fill_holes_num_views,
            debug=debug,
            verbose=verbose,
        )
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
        if verbose:
            tqdm.write(f'After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    return vertices, faces

def to_trimesh(app_rep: Gaussian, mesh: MeshExtractResult, simplify: float = 0.95, debug: bool = False,
    fill_holes: bool = True, fill_holes_max_size: float = 0.04, texture_size: int = 1024, verbose: bool = True) -> trimesh.Trimesh:
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    # mesh postprocess
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=fill_holes_max_size,
        fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)),
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=debug,
        verbose=verbose,
    )

    vertices, faces, uvs = parametrize_mesh(vertices, faces)
    # bake texture
    observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=1024, nviews=100)
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
    texture = bake_texture(
        vertices, faces, uvs,
        observations, masks, extrinsics, intrinsics,
        texture_size=texture_size, mode='opt',
        lambda_tv=0.01,
        verbose=verbose
    )
    texture = Image.fromarray(texture)

    # rotate mesh (from z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
    )
    mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    
    return mesh

SR_cache = None

def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs

def bake_vertex_colors(
    vertices: np.array,
    faces: np.array, 
    init_colors: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    projections: List[np.array] = None,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal['fast', 'opt'] = 'opt',
    lambda_smooth: float = 0,
    verbose: bool = False,
    iterations: int = 1000
):
    """
    Bake colors to mesh vertices from multiple observations using differential rasterization.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        init_colors (np.array): Initial vertex colors. Shape (V, 3).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        mode (Literal['fast', 'opt']): Mode of color baking.
        lambda_smooth (float): Weight of smoothness loss in optimization.
        verbose (bool): Whether to print progress.

    Returns:
        np.array: Vertex colors. Shape (V, 3).
    """
    vertices = torch.tensor(vertices).cuda()
    faces = torch.tensor(faces.astype(np.int32)).cuda()
    init_colors = torch.tensor(init_colors / 255.0).float().cuda()  # Convert to float [0,1]
    observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
    masks = [torch.tensor(m > 0).bool().cuda() for m in masks]
    views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
    if projections is None:
        projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).cuda(), near, far) for intr in intrinsics]

    if mode == 'fast':
        vertex_colors = init_colors.clone()  # Start from initial colors
        vertex_weights = torch.zeros(vertices.shape[0], dtype=torch.float32).cuda()
        glctx = dr.RasterizeCudaContext()

        for observation, mask, view, projection in tqdm(zip(observations, masks, views, projections), 
                                                      total=len(observations), disable=not verbose, 
                                                      desc='Color baking (fast)'):
            with torch.no_grad():
                # Transform vertices to clip space with homogeneous coordinates
                view_proj_matrix = projection @ view
                vertices_homogeneous = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)
                vertices_clip = vertices_homogeneous @ view_proj_matrix.T
                
                # Rasterize using differential renderer
                # Add batch dimension and ranges for nvdiffrast (ranges must be on CPU)
                vertices_clip = vertices_clip.unsqueeze(0)  # [1, V, 4]
                ranges = torch.tensor([[0, vertices_clip.shape[1]]], dtype=torch.int32)  # Keep ranges on CPU
                rast, _ = dr.rasterize(glctx, vertices_clip, faces, 
                                     (observation.shape[1], observation.shape[0]),
                                     ranges=ranges)
                
                # Interpolate colors
                rendered = dr.interpolate(vertex_colors[None], rast, faces)[0]
                rendered = dr.antialias(rendered, rast, vertices_clip, faces)
                
                # Update colors using valid mask
                valid_mask = mask & (rast[..., 3] > 0)  # Combine visibility and input mask
                
                # Accumulate colors and weights
                vertex_contributions = dr.interpolate(rendered[None], rast, faces, grad_db=True)[1][0]
                vertex_colors.scatter_add_(0, faces.view(-1)[:, None].expand(-1, 3), 
                                        vertex_contributions.reshape(-1, 3))
                vertex_weights.scatter_add_(0, faces.view(-1), 
                                         vertex_contributions.sum(-1).reshape(-1))

        valid_vertices = vertex_weights > 0
        vertex_colors[valid_vertices] /= vertex_weights[valid_vertices].view(-1, 1)
        
        # For invisible vertices, keep their initial colors
        vertex_colors[~valid_vertices] = init_colors[~valid_vertices]

    elif mode == 'opt':
        vertex_colors = torch.nn.Parameter(init_colors.clone())  # Initialize from existing colors
        optimizer = torch.optim.Adam([vertex_colors], betas=(0.5, 0.9), lr=1e-4)
        glctx = dr.RasterizeCudaContext()
        observations = [observations.flip(0) for observations in observations]
        masks = [m.flip(0) for m in masks]
        
        edge_indices, _, _ = utils3d.torch.compute_edges(faces)
        
        def smoothness_loss(colors):
            return torch.mean(torch.norm(
                colors[edge_indices[:, 0]] - colors[edge_indices[:, 1]], 
                dim=1
            ))

        def cosine_annealing(step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))

        total_steps = iterations
        with tqdm(total=total_steps, disable=not verbose, desc='Color baking (opt)') as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                
                idx = np.random.randint(0, len(observations))
                observation = observations[idx]
                mask = masks[idx]
                view = views[idx]
                projection = projections[idx]

                # Transform vertices to clip space with homogeneous coordinates
                view_proj_matrix = projection @ view
                vertices_homogeneous = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)
                vertices_clip = vertices_homogeneous @ view_proj_matrix.T

                # Rasterize and render
                vertices_clip = vertices_clip.unsqueeze(0)  # [1, V, 4]
                ranges = torch.tensor([[0, vertices_clip.shape[1]]], dtype=torch.int32)  # Keep ranges on CPU
                rast, _ = dr.rasterize(glctx, vertices_clip, faces, 
                                     (observation.shape[1], observation.shape[0]),
                                     ranges=ranges)
                rendered = dr.interpolate(vertex_colors[None], rast, faces)[0]
                rendered = dr.antialias(rendered, rast, vertices_clip, faces)

                # Compute loss on valid pixels
                valid_mask = mask & (rast[..., 3] > 0)
                
                color_loss = torch.nn.functional.mse_loss(
                    rendered[valid_mask],
                    observation[valid_mask.squeeze(0)]
                )
                smooth_loss = smoothness_loss(vertex_colors) if lambda_smooth > 0 else 0
                loss = color_loss + lambda_smooth * smooth_loss

                loss.backward()
                optimizer.step()

                lr = cosine_annealing(step, total_steps, 1e-2, 1e-5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

        vertex_colors = vertex_colors.detach()

    else:
        raise ValueError(f'Unknown mode: {mode}')
    
    vertex_colors = vertex_colors.clamp(0, 1).cpu().numpy()
    # vertex_colors = linear_to_srgb(vertex_colors.clamp(0, 1).cpu().numpy())
    # vertex_colors = srgb_to_linear(vertex_colors.clamp(0, 1).cpu().numpy())
    return (vertex_colors * 255).astype(np.uint8)

def srgb_to_linear(srgb):
    # Clip values to valid range
    srgb = np.clip(srgb, 0, 1)
    
    # Apply inverse sRGB transform
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4)
    )
    return linear

def linear_to_srgb(linear):
    # Clip values to valid range
    linear = np.clip(linear, 0, 1)
    
    # Apply sRGB transform
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1/2.4) - 0.055
    )
    return srgb

def bake_gs_to_mesh(
    app_rep: Union[Gaussian],
    mesh: MeshExtractResult,
    verbose: bool
) -> trimesh.Trimesh:
    # mesh = mesh.subdivide()
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    init_colors = mesh.vertex_attrs[:, :3].cpu().numpy()

    # Convert initial colors to uint8 if they're in float format
    if init_colors.dtype != np.uint8:
        init_colors = (init_colors.clip(0, 1) * 255).astype(np.uint8)
    
    # Render multiview images for color baking
    observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=2048, nviews=20)
    observations = observations['color']
    # observations = run_sr_fast(observations['color'])
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    extrinsics = [extrinsics[i].detach().cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].detach().cpu().numpy() for i in range(len(intrinsics))]

    # Bake vertex colors using initial colors
    vertex_colors = bake_vertex_colors(
        vertices, faces, init_colors,
        observations, masks, extrinsics, intrinsics,
        mode='opt',
        lambda_smooth=0.05,
        verbose=verbose
    )

    mesh.vertex_attrs[:, :3] = torch.tensor(vertex_colors / 255.0).float().cuda()
    
    return mesh


def bake_observations_to_mesh(
    observations, 
    masks,
    extrinsics, 
    intrinsics,
    projections,
    mesh: MeshExtractResult,
    verbose: bool
) -> trimesh.Trimesh:
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    init_colors = mesh.vertex_attrs[:, :3].cpu().numpy()

    # Convert initial colors to uint8 if they're in float format
    if init_colors.dtype != np.uint8:
        init_colors = (init_colors.clip(0, 1) * 255).astype(np.uint8)
        
    # Bake vertex colors using initial colors
    vertex_colors = bake_vertex_colors(
        vertices, faces, init_colors,
        observations, masks, extrinsics, intrinsics, projections,
        mode='opt',
        lambda_smooth=0.,
        verbose=verbose
    )

    mesh.vertex_attrs[:, :3] = torch.tensor(vertex_colors / 255.0).float().cuda()
    
    return mesh

def simplify_gs(
    gs: Gaussian,
    simplify: float = 0.95,
    verbose: bool = True,
):
    """
    Simplify 3D Gaussians
    NOTE: this function is not used in the current implementation for the unsatisfactory performance.
    
    Args:
        gs (Gaussian): 3D Gaussian.
        simplify (float): Ratio of Gaussians to remove in simplification.
    """
    if simplify <= 0:
        return gs
    
    # simplify
    observations, extrinsics, intrinsics = render_multiview(gs, resolution=2048, nviews=20)
    # observations = observations['color']
    observations = run_sr_fast(observations['color'])
    observations = [torch.tensor(obs / 255.0).float().cuda().permute(2, 0, 1) for obs in observations]
    
    # Following https://arxiv.org/pdf/2411.06019
    renderer = GaussianRenderer({
            "resolution": 2048,
            "near": 0.8,
            "far": 1.6,
            "ssaa": 1,
            "bg_color": (0,0,0),
        })
    new_gs = Gaussian(**gs.init_params)
    new_gs._features_dc = gs._features_dc.clone()
    new_gs._features_rest = gs._features_rest.clone() if gs._features_rest is not None else None
    new_gs._opacity = torch.nn.Parameter(gs._opacity.clone())
    new_gs._rotation = torch.nn.Parameter(gs._rotation.clone())
    new_gs._scaling = torch.nn.Parameter(gs._scaling.clone())
    new_gs._xyz = torch.nn.Parameter(gs._xyz.clone())
    
    # start_lr = [1e-4, 1e-3, 5e-3, 0.025, 1e-2]
    start_lr = [1e-4, 1e-3, 5e-3, 0.025, 1e-2]
    end_lr = [1e-6, 1e-5, 5e-5, 0.00025, 1e-4]
    optimizer = torch.optim.Adam([
        {"params": new_gs._xyz, "lr": start_lr[0]},
        {"params": new_gs._rotation, "lr": start_lr[1]},
        {"params": new_gs._scaling, "lr": start_lr[2]},
        {"params": new_gs._features_dc, "lr": start_lr[4]},
    ], lr=start_lr[0])
    
    def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return start_lr * (end_lr / start_lr) ** (step / total_steps)

    def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
    
    _zeta = new_gs.get_opacity.clone().detach().squeeze()
    _lambda = torch.zeros_like(_zeta)
    _delta = 1e-7
    _interval = 10
    num_target = int((1 - simplify) * _zeta.shape[0])
    
    with tqdm(total=2500, disable=not verbose, desc='Simplifying Gaussian') as pbar:
        for i in range(2500):
            # prune
            # if i % 100 == 0:
            #     mask = new_gs.get_opacity.squeeze() > 0.05
            #     mask = torch.nonzero(mask).squeeze()
            #     new_gs._xyz = torch.nn.Parameter(new_gs._xyz[mask])
            #     new_gs._rotation = torch.nn.Parameter(new_gs._rotation[mask])
            #     new_gs._scaling = torch.nn.Parameter(new_gs._scaling[mask])
            #     new_gs._opacity = torch.nn.Parameter(new_gs._opacity[mask])
            #     new_gs._features_dc = new_gs._features_dc[mask]
            #     new_gs._features_rest = new_gs._features_rest[mask] if new_gs._features_rest is not None else None
            #     _zeta = _zeta[mask]
            #     _lambda = _lambda[mask]
            #     # update optimizer state
            #     for param_group, new_param in zip(optimizer.param_groups, [new_gs._xyz, new_gs._rotation, new_gs._scaling, new_gs._opacity]):
            #         stored_state = optimizer.state[param_group['params'][0]]
            #         if 'exp_avg' in stored_state:
            #             stored_state['exp_avg'] = stored_state['exp_avg'][mask]
            #             stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
            #         del optimizer.state[param_group['params'][0]]
            #         param_group['params'][0] = new_param
            #         optimizer.state[param_group['params'][0]] = stored_state

            opacity = new_gs.get_opacity.squeeze()
            
            # sparisfy
            # if i % _interval == 0:
            #     _zeta = _lambda + opacity.detach()
            #     if opacity.shape[0] > num_target:
            #         index = _zeta.topk(num_target)[1]
            #         _m = torch.ones_like(_zeta, dtype=torch.bool)
            #         _m[index] = 0
            #         _zeta[_m] = 0
            #     _lambda = _lambda + opacity.detach() - _zeta
            
            # sample a random view
            view_idx = np.random.randint(len(observations))
            observation = observations[view_idx]
            extrinsic = extrinsics[view_idx]
            intrinsic = intrinsics[view_idx]
            
            color = renderer.render(new_gs, extrinsic, intrinsic)['color']
            rgb_loss = torch.nn.functional.l1_loss(color, observation)
            # loss = rgb_loss
            ssimloss = 1.0 - ssim(color.unsqueeze(0), observation.unsqueeze(0))
            loss = rgb_loss * 0.8 + ssimloss * 0.2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update lr
            for j in range(len(optimizer.param_groups)):
                optimizer.param_groups[j]['lr'] = cosine_anealing(optimizer, i, 100, start_lr[j], end_lr[j])
            
            pbar.set_postfix({'loss': loss.item(), 'num': opacity.shape[0], 'lambda': _lambda.mean().item()})
            pbar.update()
            
    new_gs._xyz = new_gs._xyz.data
    new_gs._rotation = new_gs._rotation.data
    new_gs._scaling = new_gs._scaling.data
    new_gs._opacity = new_gs._opacity.data
    
    return new_gs