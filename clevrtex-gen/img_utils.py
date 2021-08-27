import numpy as np
from PIL import Image

import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil


def cc(pil_image):
    slope = 14
    limit = 1200
    return to_pil(cca.automatic_color_equalization(from_pil(pil_image), slope, limit))

def colorcorrect(img_path, output_path):
    img = Image.open(img_path)
    cc_img = cc(img).convert('RGB')
    cc_img.save(output_path)
    return np.array(cc_img)

def load(img_path):
    return np.array(Image.open(img_path).convert('RGB'))

CMAP = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 128, 0],
    [0, 0, 255],
    [255, 255, 0],
    [141, 211, 199],
    [255, 255, 179],
    [190, 186, 218],
    [251, 128, 114],
    [128, 177, 211],
    [253, 180, 98],
    [179, 222, 105],
    [252, 205, 229],
    [217, 217, 217]])

PALIMG = Image.new('P', (16,16))
PALIMG.putpalette(CMAP.flatten().tolist() * 4)

# def post_process_img(img_path, max_num_objs, no_cc=False, masks=[]):
#     flt_path = img_path.replace('.png', '_flat.png')
#     img = Image.open(img_path)
#     # flt = Image.open(flt_path)
#     # flt = np.array(flt)[:, :, :3]
#     if not no_cc:
#         img = cc(img)
#     # extracted, n = extract_regions(flt, max_num_objs)
#     n = len(masks)
#     mask = np.zeros((img.height, img.width, n + 1), dtype=bool)
#     for i, m in enumerate(masks, start=1):
#         m = Image.open(m)
#         mask[..., i] = np.array(m).astype(bool)
#     mask[..., 0] = ~mask[..., 1:].any(-1)
#     final = mask.argmax(-1).astype(np.uint8)
#     final_out = np.array([[CMAP[v] for v in c] for c in final]).astype(np.uint8)
#     final_mask = Image.fromarray(final_out)
#     final_mask.save(flt_path)
#     objs = np.array([True] * n + [False] * (max_num_objs + 1 - n))
#     mask_path = img_path.replace('.png', '.npz')
#     np.savez_compressed(mask_path, mask=mask, visibility=objs, image=np.array(img))
#     if not no_cc:
#         new_img_path = img_path.replace('.png', '_cc.jpg')
#         img.convert('RGB').save(new_img_path)
#         return mask_path, new_img_path
#     return mask_path, img_path



def collect_object_masks(masks, max_num_objects, mask_out_path):
    mask_list = [np.array(Image.open(m)).astype(bool) for m in masks]
    for _ in range(len(mask_list), max_num_objects+1):
        mask_list.append(np.zeros_like(mask_list[0], dtype=np.bool))
    mask_list.insert(0, np.zeros_like(mask_list[0], dtype=np.bool))
    mask = np.array(mask_list, dtype=np.bool)
    mask = np.moveaxis(mask, 0, -1)
    mask[..., 0] = ~mask[..., 1:].any(-1)

    final = mask.argmax(-1).astype(np.uint8)

    final_mask = Image.fromarray(final).convert('L')
    final_mask.load()
    im = final_mask.im.convert('P', 0, PALIMG.im)
    final_mask = final_mask._new(im)
    # final_out = np.array([[CMAP[v] for v in c] for c in final]).astype(np.uint8)
    # final_mask = Image.fromarray(final_out)
    final_mask.save(mask_out_path)
    return mask
