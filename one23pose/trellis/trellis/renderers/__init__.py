import importlib

__attributes = {
    'GSplatRenderer': 'gsplat_renderer',
    'GaussianRenderer': 'gaussian_render',
    'MeshRenderer': 'mesh_renderer',
    'OctreeRenderer': 'octree_renderer'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .mesh_renderer import MeshRenderer
    from .gsplat_renderer import GSplatRenderer
    from .gaussian_render import GaussianRenderer
    from .octree_renderer import OctreeRenderer