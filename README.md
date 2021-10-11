# ClevrTex

This repository contains dataset generation code for ClevrTex benchmark from paper:
**[ClevrTex: A Texture-Rich Benchmark for Unsupervised Multi-Object Segmentation](https://www.robots.ox.ac.uk/~vgg/research/clevrtex)**

## Requirements

The follwing preparation steps are required to generate the dataset.
1. Setting up blender
2. Setting up python
3. Setting up textures and materials

#### Blender
We used blender 2.92.3 for rendering. Newer versions are untested but _should_ work at least up to a minor bump. One might download it from [Blender website](https://www.blender.org) and follow installation instructions process as normal
then skip to the final step. Or simply execute this (will set up blender in /usr/local/blender):
```
mkdir /usr/local/blender && \
curl -SL "http://mirror.cs.umn.edu/blender.org/release/Blender2.92/blender-2.92.0-linux64.tar.xz" -o blender.tar.xz && \
tar -xvf blender.tar.xz -C /usr/local/blender --strip-components=1 && \
rm blender.tar.xz && ln -s /usr/local/blender/blender /usr/local/bin/blender
```
Since we use "system interpreter" (see intructions bellow to set up a compatible one) for Blender headless mode, remove
python that comes pre-packaged.
```bash
rm -rf /usr/local/blender/2.92/python
```

#### Python
One needs to set up python with required libraries and with correct version. Blender uses python 3.7 
(older or newer version will not work). For simplicty, use conda:
```bash
conda env create -f env.yaml
```
When invoking Blender use (assumes the appropriate env was named `p37`) :
```bash
PYTHONPATH=~/miniconda3/envs/p37/bin/python \
PYTHONHOME=~/miniconda3/envs/p37 \
blender --background --python-use-system-env --python generate.py -- <args>
```

#### Textures
The final piece is to set up source assets for rendering, namely the materials. 
Briefly, the textures used to create the materials are copyrighted by Poliigon Pty Ltd.
Textures used in the ClevrTex dataset are freely availble (at the time of writing) and should be
downloaded from www.poliigon.com  (download metalness workflow for matalics). Please check MATERIALS.md for full list.

Download appropriate textures and place them into `data/materials/textures` and `data/outd_materials/textures`. Note, the textures should be in the directory not in subfolders. 
We include .blend files for materials which have been stripped of the original textures (due to licensing restrictions) but contain the settings adjustments made.
Skip the following instructions if working with existing .blend files.

##### To add new materials:
The following process needs to be applied for each new material. Consider using [addon](https://help.poliigon.com/en/articles/2540839-poliigon-material-converter-addon-for-blender) provided by Poliigon.
1. Import materials textures as per addon's instructions.
2. Open the material in question in node editor in Blender.
3. Create a new node group of all nodes except the output node (yes this will nest the groups, it is intentional).
We rely on the trick identified by Johnson et al. in the original 
   CLEVR script where Blender seems to 
   copy-by-value node trees, which makes it trivial to create 
   duplicate materials in the scene.
4. Connect any inputs of interest to the group inputs. Crucially, check that **Scale** and **Displacement Strength** are available as inputs.
   The sampling script will pass these in to ensure that background/objects have correct scale adjustements to ensure level of details does not disappear between small objects and large background.
   Check that outputs have been connected to Shader output nodes (should have happended automatically).
5. Ensure that the materials look good with other parameters. Consider including additional logic nodes to e.g. scaling, and displacement parameters. 
   Materials have Random \in [0, 1] number passed to them as input (if available), if one needs to randomise aspects of the material. 
    - (Optional) Render the materials to see how they would look in the output. Repeat until desired look is acheived.
6. Ensure the node group is named identically to the material and then save it as your-node-group-name.blend.

This is unfortunatelly a manual process to ensure all textures look good that usually involves several test render per texture.

#### Debugging textures
To ensure the textures are found and look good, consider trying with a single texture first (to save time).
To scan for errors and see how the end result might look like, consider using `--test_scan` option in the generation script.*
In addition, consider `--blendfiles` option to save blender scene after rendering for manual inspection. 


## Generating
To generate the dataset run the following (will produce a LOCAL_debug_000001.png example):
```bash
cd clevrtex-gen
 ./local_test.bash
```

Otherwise, please see arguments available to customise the rendering. Dataset variants can be recreated using appropriate 
`<variant>.json` files.

# Using ClevrTex
See project page for download links for CLEVRTEX.
`clevrtex_eval.py` file contains dataloading logic to for convenient access to CLEVRTEX data.
Consider
```python
from clevrtex_eval import CLEVRTEX, collate_fn

clevrtex = CLEVRTEX(
    'path-to-downloaded-data', # Untar'ed
    dataset_variant='full', # 'full' for main CLEVRTEX, 'outd' for OOD, 'pbg','vbg','grassbg','camo' for variants.
    split='train',
    crop=True,
    resize=(128, 128),
    return_metadata=True # Useful only for evaluation, wastes time on I/O otherwise 
)
# Use collate_fn to handle metadata batching
dataloader = torch.utils.data.DataLoader(clevrtex, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
```

# Evaluation
See `CLEVRTEX_Evaluator` in `clevrtex_eval.py`. It implements all the utilities needed.

### CLEVR
This dataset builds upon
**[CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](http://cs.stanford.edu/people/jcjohns/clevr/)**
 <br>
 <a href='http://cs.stanford.edu/people/jcjohns/'>Justin Johnson</a>,
 <a href='http://home.bharathh.info/'>Bharath Hariharan</a>,
 <a href='https://lvdmaaten.github.io/'>Laurens van der Maaten</a>,
 <a href='http://vision.stanford.edu/feifeili/'>Fei-Fei Li</a>,
 <a href='http://larryzitnick.org/'>Larry Zitnick</a>,
 <a href='http://www.rossgirshick.info/'>Ross Girshick</a>
 <br>
 presented at [CVPR 2017](http://cvpr2017.thecvf.com/), code available at https://github.com/facebookresearch/clevr-dataset-gen

In particular we use a method for computing cardinal directions from CLEVR.
See the original licence included in the clevr_qa.py file.

# BibTeX
If you use ClevrTex dataset or generation code consider citing:
```
BiBTeX coming soon...
```
