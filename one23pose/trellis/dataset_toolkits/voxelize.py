import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import open3d as o3d
import utils3d
from tqdm import tqdm

def _voxelize(file, sha256, output_dir):
    mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'render_all_eevee_1024_150views_evenbg', sha256, 'mesh.ply'))
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    utils3d.io.write_ply(os.path.join(output_dir, 'voxels', f'{sha256}.ply'), vertices)
    return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=None)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # get file list
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata_all_eevee_1024_150views_evenbg'))
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' not in metadata.columns:
            raise ValueError('metadata.csv does not have "rendered" column, please run "build_metadata.py" first')
        metadata = metadata[metadata['rendered'] == False]
        if 'voxelized' in metadata.columns:
            metadata = metadata[metadata['voxelized'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    os.makedirs(os.path.join(opt.output_dir, 'voxels'), exist_ok=True)
    # filter out objects that are already processed
    import concurrent.futures

    def process_sha256(sha256):
        path = os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')
        if os.path.exists(path):
            try:
                pts = utils3d.io.read_ply(path)[0]
                return sha256, len(pts)
            except Exception:
                return None
        return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=opt.max_workers or 16) as executor:
        futures = {executor.submit(process_sha256, sha): sha for sha in metadata['sha256'].values}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Checking cached voxels"):
            result = future.result()
            if result is not None:
                results.append(result)

    # update records and filter out processed sha256 from metadata
    for sha256, num_voxels in results:
        records.append({'sha256': sha256, 'voxelized': True, 'num_voxels': num_voxels})
    processed_set = {rec['sha256'] for rec in records}
    metadata = metadata[~metadata['sha256'].isin(processed_set)]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_voxelize, output_dir=opt.output_dir)
    voxelized = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Voxelizing')
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(os.path.join(opt.output_dir, f'voxelized_{opt.rank}.csv'), index=False)
