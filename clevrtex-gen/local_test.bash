export PYTHONPATH=~/miniconda3/envs/p37/bin/python
export PYTHONHOME=~/miniconda3/envs/p37
~/blender/blender --background \
     --python-use-system-env --python generate.py -- \
     --render_tile_size 64 \
     --width 320 --height 240 \
		 --start_idx 0 \
		 --filename_prefix LOCAL \
		 --variant debug \
		 --output_dir ../output/ \
		 --properties_json data/full.json \
		 --shape_dir data/shapes \
		 --material_dir data/materials \
		 --blendfiles \
		 --num_images 1
