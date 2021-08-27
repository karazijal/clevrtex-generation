import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


class DatasetReadError(ValueError):
    pass


class CLEVRTEX:
    ccrop_frac = 0.8
    splits = {
        'test': (0., 0.1),
        'val': (0.1, 0.2),
        'train': (0.2, 1.)
    }
    shape = (3, 240, 320)
    variants = {'full', 'pbg', 'vbg', 'grassbg', 'camo', 'outd'}

    def _index_with_bias_and_limit(self, idx):
        if idx >= 0:
            idx += self.bias
            if idx >= self.limit:
                raise IndexError()
        else:
            idx = self.limit + idx
            if idx < self.bias:
                raise IndexError()
        return idx

    def _reindex(self):
        print(f'Indexing {self.basepath}')

        img_index = {}
        msk_index = {}
        met_index = {}

        prefix = f"CLEVRTEX_{self.dataset_variant}_"

        img_suffix = ".png"
        msk_suffix = "_flat.png"
        met_suffix = ".json"

        _max = 0
        for img_path in self.basepath.glob(f'**/{prefix}??????{img_suffix}'):
            indstr = img_path.name.replace(prefix, '').replace(img_suffix, '')
            msk_path = img_path.parent / f"{prefix}{indstr}{msk_suffix}"
            met_path = img_path.parent / f"{prefix}{indstr}{met_suffix}"
            indstr_stripped = indstr.lstrip('0')
            if indstr_stripped:
                ind = int(indstr)
            else:
                ind = 0
            if ind > _max:
                _max = ind

            if not msk_path.exists():
                raise DatasetReadError(f"Missing {msk_suffix.name}")

            if ind in img_index:
                raise DatasetReadError(f"Duplica {ind}")

            img_index[ind] = img_path
            msk_index[ind] = msk_path
            if self.return_metadata:
                if not met_path.exists():
                    raise DatasetReadError(f"Missing {met_path.name}")
                met_index[ind] = met_path
            else:
                met_index[ind] = None

        if len(img_index) == 0:
            raise DatasetReadError(f"No values found")
        missing = [i for i in range(0, _max) if i not in img_index]
        if missing:
            raise DatasetReadError(f"Missing images numbers {missing}")

        return img_index, msk_index, met_index

    def _variant_subfolder(self):
        return f"clevrtex_{self.dataset_variant.lower()}"

    def __init__(self,
                 path: Path,
                 dataset_variant='full',
                 split='train',
                 crop=True,
                 resize=(128, 128),
                 return_metadata=True):
        self.return_metadata = return_metadata
        self.crop = crop
        self.resize = resize
        if dataset_variant not in self.variants:
            raise DatasetReadError(f"Unknown variant {dataset_variant}; [{', '.join(self.variants)}] available ")

        if split not in self.splits:
            raise DatasetReadError(f"Unknown split {split}; [{', '.join(self.splits)}] available ")
        if dataset_variant == 'outd':
            # No dataset splits in
            split = None

        self.dataset_variant = dataset_variant
        self.split = split

        self.basepath = Path(path)
        if not self.basepath.exists():
            raise DatasetReadError()
        sub_fold = self._variant_subfolder()
        if self.basepath.name != sub_fold:
            self.basepath = self.basepath / sub_fold
        #         try:
        #             with (self.basepath / 'manifest_ind.json').open('r') as inf:
        #                 self.index = json.load(inf)
        #         except (json.JSONDecodeError, IOError, FileNotFoundError):
        self.index, self.mask_index, self.metadata_index = self._reindex()

        print(f"Sourced {dataset_variant} ({split}) from {self.basepath}")

        bias, limit = self.splits.get(split, (0., 1.))
        if isinstance(bias, float):
            bias = int(bias * len(self.index))
        if isinstance(limit, float):
            limit = int(limit * len(self.index))
        self.limit = limit
        self.bias = bias

    def _format_metadata(self, meta):
        """
        Drop unimportanat, unsued or incorrect data from metadata.
        Data may become incorrect due to transformations,
        such as cropping and resizing would make pixel coordinates incorrect.
        Furthermore, only VBG dataset has color assigned to objects, we delete the value for others.
        """
        objs = []
        for obj in meta['objects']:
            o = {
                'material': obj['material'],
                'shape': obj['shape'],
                'size': obj['size'],
                'rotation': obj['rotation'],
            }
            if self.dataset_variant == 'vbg':
                o['color'] = obj['color']
            objs.append(o)
        return {
            'ground_material': meta['ground_material'],
            'objects': objs
        }

    def __len__(self):
        return self.limit - self.bias

    def __getitem__(self, ind):
        ind = self._index_with_bias_and_limit(ind)

        img = Image.open(self.index[ind])
        msk = Image.open(self.mask_index[ind])

        if self.crop:
            crop_size = int(0.8 * float(min(img.width, img.height)))
            img = img.crop(((img.width - crop_size) // 2,
                            (img.height - crop_size) // 2,
                            (img.width + crop_size) // 2,
                            (img.height + crop_size) // 2))
            msk = msk.crop(((msk.width - crop_size) // 2,
                            (msk.height - crop_size) // 2,
                            (msk.width + crop_size) // 2,
                            (msk.height + crop_size) // 2))
        if self.resize:
            img = img.resize(self.resize, resample=Image.BILINEAR)
            msk = msk.resize(self.resize, resample=Image.NEAREST)

        img = Ft.to_tensor(np.array(img)[..., :3])
        msk = torch.from_numpy(np.array(msk))[None]

        ret = (ind, img, msk)

        if self.return_metadata:
            with self.metadata_index[ind].open('r') as inf:
                meta = json.load(inf)
            ret = (ind, img, msk, self._format_metadata(meta))

        return ret


def collate_fn(batch):
    return (
        *torch.utils.data._utils.collate.default_collate([(b[0], b[1], b[2]) for b in batch]), [b[3] for b in batch])


class RunningMean:
    def __init__(self):
        self.v = 0.
        self.n = 0

    def update(self, v, n=1):
        self.v += v * n
        self.n += n

    def value(self):
        if self.n:
            return self.v / (self.n)
        else:
            return float('nan')

    def __str__(self):
        return str(self.value())


class CLEVRTEX_Evaluator:
    def __init__(self, masks_have_background=True):
        self.masks_have_background = masks_have_background
        self.stats = defaultdict(RunningMean)
        self.tags = defaultdict(lambda: defaultdict(lambda: defaultdict(RunningMean)))

    def ari(self, pred_mask, true_mask, skip_0=False):
        B = pred_mask.shape[0]
        pm = pred_mask.argmax(axis=1).squeeze().view(B, -1).cpu().detach().numpy()
        tm = true_mask.argmax(axis=1).squeeze().view(B, -1).cpu().detach().numpy()
        aris = []
        for bi in range(B):
            t = tm[bi]
            p = pm[bi]
            if skip_0:
                p = p[t > 0]
                t = t[t > 0]
            ari_score = adjusted_rand_score(t, p)
            if ari_score != ari_score:
                print(f'NaN at bi')
            aris.append(ari_score)
        aris = torch.tensor(np.array(aris), device=pred_mask.device)
        return aris

    def msc(self, pred_mask, true_mask):
        B = pred_mask.shape[0]
        bpm = pred_mask.argmax(axis=1).squeeze()
        btm = true_mask.argmax(axis=1).squeeze()
        covering = torch.zeros(B, device=pred_mask.device, dtype=torch.float)
        for bi in range(B):
            score = 0.
            norms = 0.
            for ti in range(btm[bi].max()):
                tm = btm[bi] == ti
                if not torch.any(tm): continue
                iou_max = 0.
                for pi in range(bpm[bi].max()):
                    pm = bpm[bi] == pi
                    if not torch.any(pm): continue
                    iou = (tm & pm).to(torch.float).sum() / (tm | pm).to(torch.float).sum()
                    if iou > iou_max:
                        iou_max = iou
                r = tm.to(torch.float).sum()
                score += r * iou_max
                norms += r
            covering[bi] = score / norms
        return covering






    def reindex(self, tensor, reindex_tensor, dim=1):
        """
        Reindexes tensor along <dim> using reindex_tensor.
        Effectivelly permutes <dim> for each dimensions <dim based on values in reindex_tensor
        """
        # add dims at the end to match tensor dims.
        alignment_index = reindex_tensor.view(*reindex_tensor.shape,
                                              *([1] * (tensor.dim() - reindex_tensor.dim())))
        return torch.gather(tensor, dim, alignment_index.expand_as(tensor))

    def ious_alignment(self, pred_masks, true_masks):
        tspec = dict(device=pred_masks.device)
        iou_matrix = torch.zeros(pred_masks.shape[0], pred_masks.shape[1], true_masks.shape[1], **tspec)

        true_masks_sums = true_masks.sum((-1, -2, -3))
        pred_masks_sums = pred_masks.sum((-1, -2, -3))

        pred_masks = pred_masks.to(torch.bool)
        true_masks = true_masks.to(torch.bool)

        # Fill IoU row-wise
        for pi in range(pred_masks.shape[1]):
            # Intersection against all cols
            # pandt = (pred_masks[:, pi:pi + 1] * true_masks).sum((-1, -2, -3))
            pandt = (pred_masks[:, pi:pi + 1] & true_masks).to(torch.float).sum((-1, -2, -3))
            # Union against all colls
            # port = pred_masks_sums[:, pi:pi + 1] + true_masks_sums
            port = (pred_masks[:, pi:pi + 1] | true_masks).to(torch.float).sum((-1, -2, -3))
            iou_matrix[:, pi] = pandt / port
            iou_matrix[pred_masks_sums[:, pi] == 0., pi] = 0.

        for ti in range(true_masks.shape[1]):
            iou_matrix[true_masks_sums[:, ti] == 0., :, ti] = 0.

        # NaNs, Inf might come from empty masks (sums are 0, such as on empty masks)
        # Set them to 0. as there are no intersections here and we should not reindex
        iou_matrix = torch.nan_to_num(iou_matrix, nan=0., posinf=0., neginf=0.)

        cost_matrix = iou_matrix.cpu().detach().numpy()
        ious = np.zeros(pred_masks.shape[:2])
        pred_inds = np.zeros(pred_masks.shape[:2], dtype=int)
        for bi in range(cost_matrix.shape[0]):
            true_ind, pred_ind = linear_sum_assignment(cost_matrix[bi].T, maximize=True)
            cost_matrix[bi].T[:, pred_ind].argmax(1)  # Gives which true mask is best for EACH predicted
            ious[bi] = cost_matrix[bi].T[true_ind, pred_ind]
            pred_inds[bi] = pred_ind

        ious = torch.from_numpy(ious).to(pred_masks.device)
        pred_inds = torch.from_numpy(pred_inds).to(pred_masks.device)
        return pred_inds, ious, iou_matrix

    def add_statistic(self, name, value, **tags):
        n = 1
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach()
            if len(value.shape):
                n = value.shape[0]
                value = torch.mean(value)
            value = value.item()
        self.stats[name].update(value, n)
        for k, v in tags.items():
            self.tags[name][k][v].update(value, n)

    def statistic(self, name, tag=None):
        if tag is None:
            return self.stats[name].value()
        r = [(k, rm.value()) for k, rm in self.tags[name][tag].items()]
        r = sorted(r, key=lambda x: x[1])
        return r

    @torch.no_grad()
    def update(self,
               pred_image,
               pred_masks,
               true_image,
               true_masks,
               true_metadata=None):
        assert len(pred_image.shape) == 4, "Images should be in (B, C, H, W) shape"

        # TODO: types
        # Coerce pred_masks into known form
        assert 4 <= len(pred_masks.shape) <= 5, "Masks shoudl be in (B, K, 1, H, W) shape"
        pred_masks = pred_masks.view(pred_image.shape[0], -1, 1, *pred_image.shape[-2:])
        total_pred_masks = pred_masks.sum(1, keepdims=True)
        #         assert torch.any(total_pred_masks > 1), "Predicted masks sum out to more than 1."
        if not self.masks_have_background:
            # Some models predict only foreground masks.
            # For convenienve we calculate background masks.
            pred_masks = torch.cat([1. - total_pred_masks, pred_masks], dim=1)

        # Decide the masks Should we effectivelly threshold them?
        K = pred_masks.shape[1]
        pred_masks = pred_masks.argmax(dim=1)
        pred_masks = (pred_masks.unsqueeze(1) == torch.arange(K, device=pred_masks.device).view(1, -1, 1, 1, 1)).to(
            torch.float)
        # Coerce true_Masks into known form
        if len(true_masks.shape) == 4:
            if true_masks.shape[1] == 1:
                # Need to expand into masks
                true_masks = (true_masks.unsqueeze(1) == torch.arange(max(true_masks.max() + 1, pred_masks.shape[1]),
                                                                      device=true_masks.device).view(1, -1, 1, 1,
                                                                                                     1)).to(
                    pred_image.dtype)
            else:
                true_masks = true_masks.unsqueeze(2)
        true_masks = true_masks.view(pred_image.shape[0], -1, 1, *pred_image.shape[-2:])

        K = max(true_masks.shape[1], pred_masks.shape[1])
        if true_masks.shape[1] < K:
            true_masks = torch.cat([true_masks, true_masks.new_zeros(true_masks.shape[0], K - true_masks.shape[1], 1,
                                                                     *true_masks.shape[-2:])], dim=1)
        if pred_masks.shape[1] < K:
            pred_masks = torch.cat([pred_masks, pred_masks.new_zeros(pred_masks.shape[0], K - pred_masks.shape[1], 1,
                                                                     *pred_masks.shape[-2:])], dim=1)


        mse = F.mse_loss(pred_image, true_image, reduction='none').sum((1, 2, 3))
        self.add_statistic('MSE', mse)

        # If argmax above, these masks are either 0 or 1
        pred_count = (pred_masks >= 0.5).any(-1).any(-1).any(-1).to(torch.float).sum(-1)  # shape: (B,)
        true_count = (true_masks >= 0.5).any(-1).any(-1).any(-1).to(torch.float).sum(-1)  # shape: (B,)
        accuracy = (true_count == pred_count).to(torch.float)
        self.add_statistic('acc', accuracy)

        pred_reindex, ious, _ = self.ious_alignment(pred_masks, true_masks)
        pred_masks = self.reindex(pred_masks, pred_reindex, dim=1)

        truem = true_masks.any(-1).any(-1).any(-1)
        predm = pred_masks.any(-1).any(-1).any(-1)

        vism = truem | predm
        num_pairs = vism.to(torch.float).sum(-1)

        # mIoU
        mIoU = ious.sum(-1) / num_pairs
        mIoU_fg = ious[:, 1:].sum(-1) / (num_pairs - 1)  # do not consider the background
        mIoU_gt = ious.sum(-1) / truem.to(torch.float).sum(-1)

        self.add_statistic('mIoU', mIoU)
        self.add_statistic('mIoU_fg', mIoU_fg)
        self.add_statistic('mIoU_gt', mIoU_gt)

        msc = self.msc(pred_masks, true_masks)
        self.add_statistic('mSC', msc)

        # DICE
        dices = 2 * (pred_masks * true_masks).sum((-3, -2, -1)) / (
                pred_masks.sum((-3, -2, -1)) + true_masks.sum((-3, -2, -1)))
        dices = torch.nan_to_num(dices, nan=0., posinf=0.)  # if there were any empties, they now have 0. DICE

        dice = dices.sum(-1) / num_pairs
        dice_fg = dices[:, 1:].sum(-1) / (num_pairs - 1)
        self.add_statistic('DICE', dice)
        self.add_statistic('DICE_FG', dice_fg)

        # ARI
        ari = self.ari(pred_masks, true_masks)
        ari_fg = self.ari(pred_masks, true_masks, skip_0=True)
        if torch.any(torch.isnan(ari_fg)):
            print('NaN ari_fg')
        if torch.any(torch.isinf(ari_fg)):
            print('Inf ari_fg')
        self.add_statistic('ARI', ari)
        self.add_statistic('ARI_FG', ari_fg)

        # mAP --?

        if true_metadata is not None:
            smses = F.mse_loss(pred_image[:, None] * true_masks,
                               true_image[:, None] * true_masks, reduction='none').sum((-1, -2, -3))

            for bi, meta in enumerate(true_metadata):
                # ground
                self.add_statistic('ground_mse', smses[bi, 0], ground_material=meta['ground_material'])
                self.add_statistic('ground_iou', ious[bi, 0], ground_material=meta['ground_material'])

                for i, obj in enumerate(meta['objects']):
                    tags = {k: v for k, v in obj.items() if k != 'rotation'}
                    if truem[bi, i + 1]:
                        self.add_statistic('obj_mse', smses[bi, i + 1], **tags)
                        self.add_statistic('obj_iou', ious[bi, i + 1], **tags)
                        # Maybe number of components?
        return pred_masks, true_masks
