import random
import json
from pathlib import Path


class SamplingConfig:
    def __load_shape_conf(self, shape_conf):
        if isinstance(shape_conf, str):
            shape_path = shape_conf
            shape_mats = [m for m in self.materials]
        elif isinstance(shape_conf, dict):
            shape_path = shape_conf['name']
            if 'material' in shape_conf:
                shape_mats = self.__load_material_conf(shape_conf['material'])
            else:
                shape_mats = [m for m in self.materials]
        else:
            raise ValueError(f"Unknonw shape_conf: {shape_conf}")
        return shape_path, shape_mats

    def __load_material_conf(self, mat_conf):
        if isinstance(mat_conf, str):
            if mat_conf in self.mat_map:
                return [mat_conf]
            else:
                raise ValueError(f"Unknonw material {mat_conf}")
        elif isinstance(mat_conf, dict):
            if 'include' in mat_conf:
                to_return = [m for m in mat_conf['include'] if m in self.mat_map]
            elif 'exclude' in mat_conf:
                blacklist = {m for m in mat_conf['exclude']}
                to_return = [m for m in self.mat_map if m not in blacklist]
            else:
                raise ValueError(f"Unknown {mat_conf}")
            if len(to_return) > 0:
                return to_return
            raise ValueError(f"No known materials in {mat_conf}")

    def __init__(self, conf_path, material_dir, shape_dir, scan_dirs=False, same_material=False):
        conf_path = Path(conf_path)
        material_dir = Path(material_dir)
        self.same_material = same_material
        # print(conf_path)
        with conf_path.open('r') as inf:
            self.conf = json.load(inf)

        # load materials
        self.materials = []
        for mat_name, name in self.conf.get('materials', {}).items():
            # utils.load_material(name, material_dir)
            self.materials.append((mat_name, name))
        if len(self.materials) == 0 or scan_dirs:
            self.materials = []
            for mp in material_dir.glob('*.blend'):
                # name = mp.stem
                # utils.load_material(name, str(material_dir))
                self.materials.append((mp.stem.lower(), mp))
        self.mat_map = dict(self.materials)
        # print([m for m in self.mat_map])

        # load shapes
        self.shape_dir = shape_dir
        self.shapes = []
        self.shape_map = {}
        for shape_name, shape_conf in self.conf.get('shapes', {}).items():
            shape_path, shape_mats = self.__load_shape_conf(shape_conf)
            if not shape_path.endswith('.blend'):
                shape_path += '.blend'
            shape_path = Path(shape_dir) / shape_path
            if not shape_path.exists() or not shape_path.is_file():
                raise ValueError(f"Shape {shape_name} cannot be found at {shape_path}")
            self.shapes.append((shape_name, shape_path))

            self.shape_map[shape_name] = shape_mats

        self.colors = [(c, v) for c,v in self.conf['colors'].items()]
        self.sizes = [(c, v) for c,v in self.conf['sizes'].items()]

        if 'ground' in self.conf:
            if 'material' in self.conf['ground']:
                self.ground_materials = self.__load_material_conf(self.conf['ground']['material'])
            else:
                self.ground_materials = [m for m in self.materials]
        else:
            self.ground_materials = ['tabularasa']

        if self.same_material:
            self._material = random.choice(self.ground_materials)
            n, v = random.choice(self.colors)
            self._color = n, [*[c/255. for c in v], 0]



    def sample_shape(self):
        return random.choice(self.shapes)

    def sample_material_for_shape(self, shape):
        if self.same_material:
            return self._material, self.mat_map[self._material]
        mat = random.choice(self.shape_map[shape])
        return mat, self.mat_map[mat]

    @property
    def mats(self):
        yield from self.mat_map.items()

    def sample_color_for_shape(self, shape):
        if self.same_material:
            return self._color
        n, v = random.choice(self.colors)
        return n, [*[c/255. for c in v], 0]

    def sample_ground_mat(self):
        if self.same_material:
            return self._material, self.mat_map[self._material]
        mat = random.choice(self.ground_materials)
        return mat, self.mat_map[mat]

    def sample_size(self):
        return random.choice(self.sizes)

