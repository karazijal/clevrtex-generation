# ClevrTex

This repository contains dataset generation code for ClevrTex benchmark from paper:
**[ClevrTex: A Texture-Rich Benchmark for Unsupervised Multi-Object Segmentation](https://www.robots.ox.ac.uk/~vgg/research/clevrtex)**.
For experiment code, see [here](https://github.com/karazijal/clevrtex).

# Version 2

Due to some changes of the T&Cs of the vendor we have previously obtained the materials from, it might no longer be possible to obtain and use the original materials in AI-related applications. We have thus compiled a new library of materials using textures from [Polyhaven](https://polyhaven.com/license), [ambientCG](https://ambientcg.com), and [Sharetextures.com](https://www.sharetextures.com/p/license), available under CC0 licenses. 

The new material library is available [here](https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtexv2_materials.tar.gz) for the main ClevrTex dataset, and [here](https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtexv2_outd_materials.tar.gz) for the OOD. Simply place the materails in the `data/materials` and `data/outd_materials` folders, respectively, and follow the instructions below to generate the dataset.


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
@inproceedings{karazija2021clevrtex,
  author =        {Laurynas Karazija and Iro Laina and
                   Christian Rupprecht},
  booktitle =     {Thirty-fifth Conference on Neural Information
                   Processing Systems Datasets and Benchmarks Track},
  title =         {{C}levr{T}ex: {A} {T}exture-{R}ich {B}enchmark for {U}nsupervised
                   {M}ulti-{O}bject {S}egmentation},
  year =          {2021},
}
```
