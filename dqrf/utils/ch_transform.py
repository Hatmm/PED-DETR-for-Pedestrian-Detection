# ------------------------------------------------------------------------
# Modified by Li Chuming
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import random
import numpy as np
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from dqrf.utils.box_ops import box_xyxy_to_cxcywh



def crop(image, target, region):
    # print(region)
    # print(image.size)
    # import pdb
    # pdb.set_trace()
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area"]

    if "boxes" in target:
        boxes = target["boxes"]
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = cropped_boxes.reshape(-1, 2, 2)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "vboxes" in target:
        boxes = target["vboxes"]
        max_size = torch.as_tensor([w, h, w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes, max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["vboxes"] = cropped_boxes
        fields.append("vboxes")

    if "iboxes" in target:
        target["iboxes"] = target["iboxes"] - torch.as_tensor([j, i, j, i])

    # remove elements for which the boxes or masks that have zero area
    if "vboxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "vboxes" in target:
            cropped_boxes = target['vboxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    # remove elements for which the boxes or masks that have zero area
    if "iboxes" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        cropped_boxes = target['iboxes'].reshape(-1, 2, 2)
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        target["iboxes"] = target["iboxes"][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "vboxes" in target:
        boxes = target["vboxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["vboxes"] = boxes

    if "iboxes" in target:
        boxes = target["iboxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["iboxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    fields = ["labels", "area"]

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
        fields.append("boxes")

    if "vboxes" in target:
        boxes = target["vboxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["vboxes"] = scaled_boxes
        fields.append("vboxes")

    if "iboxes" in target:
        boxes = target["iboxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["iboxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])


    if "vboxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "vboxes" in target:
            cropped_boxes = target['vboxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :] >= 4, dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        if "iboxes" in target:
            target["iboxes"] = torch.cat([target["iboxes"], target["vboxes"][~keep]], dim=0)

        for field in fields:
            target[field] = target[field][keep]

    return rescaled_image, target



class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)

class RandomCropCH(object): #### for crowdhuman
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size
        # self.size = size

    def __call__(self, img, target):
        # ensure h > th  and w > tw
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))

        region = T.RandomCrop.get_params(img, [h, w])#self.size)

        boxes = target["vboxes"]

        if len(boxes) > 0:
            # i, j are the offset -> i = random.randint(0, h - th) with h org_size, th output_size
            # gives starting point on i and j and w h gives length
            i, j, h, w = region
            # i1 starting pt
            # i2 end pt
            i1, i2, j1, j2 = int(i), int(i + h - 1), int(j), int(j + w - 1)
            w, h = img.size
            w, h = int(w), int(h)
            # clamp starting pt and end pt according the img size
            # - 1 to count for that fact that we start at 0, these i are used for indexing
            i1, i2, j1, j2 = max(i1, 0), min(i2, h - 1), max(j1, 0), min(j2, w - 1)
            # [w], [h]
            x_empty, y_empty = torch.zeros(w), torch.zeros(h)
            boxes = torch.min(boxes, torch.as_tensor([w-2, h-2, w-2, h-2]).float())
            boxes = boxes.clamp(min=1).int()
            for box in boxes:
                x1, y1, x2, y2 = box
                cx, cy, wx, wy = (x1 + x2) / 2., (y1 + y2) / 2., (x2 - x1) / 2., (y2 - y1) / 2.
                x1, x2, y1, y2 = int(cx - 0.9 * wx), int(cx + 0.9 * wx), int(cy - 0.9 * wy), int(cy + 0.9 * wy)
                for i in range(x1, x2+1):
                    x_empty[i] = 1
                for i in range(y1, y2+1):
                    y_empty[i] = 1
            validx = np.where(x_empty==0)[0]
            validy = np.where(y_empty==0)[0]
            i1 = validy[np.where(validy<=i1)[0][-1]]
            i2 = validy[np.where(validy>=i2)[0][0]]
            j1 = validx[np.where(validx<=j1)[0][-1]]
            j2 = validx[np.where(validx>=j2)[0][0]]
            region = (i1, j1, i2 - i1 + 1, j2 - j1 + 1)

        return crop(img, target, region)

class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)





class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):

    def __call__(self, image, target=None):
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "vboxes" in target:
            boxes = target["vboxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["vboxes"] = boxes
        if "iboxes" in target:
            boxes = target["iboxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["iboxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
