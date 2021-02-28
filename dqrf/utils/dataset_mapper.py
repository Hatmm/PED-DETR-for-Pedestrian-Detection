#
# Modified by Matthieu Lin
# Contact: linmatthieu@gmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import logging

import numpy as np
import torch
import torchvision.transforms.functional as F

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import dqrf.utils.ch_transform as CH_T
from detectron2.structures import Boxes, Instances, BoxMode
from dqrf.utils.utils import ImageMeta, FileSystemPILReader
from fvcore.transforms.transform import Transform
from typing import Dict

__all__ = ["DqrfDatasetMapper"]

# class ChAugInput(T.AugInput):
#     def __init__(self, image: np.ndarray, target: Dict):
#         self.image = image
#         self.targets = target
#
#     def transform(self, tfm: Transform) -> None:
#         self.image = tfm.apply_image(self.image, self.targets)
#         self.targets = tfm.apply_box(self.image, self.targets)

def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger("detectron2.data.dataset_mapper")
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens

# Modified from DETR (https://github.com/facebookresearch/detr)
class DqrfDatasetMapper:
    """
    A callable which takes a datasets dict in Detectron2 Dataset format,
    and map it into a format used by DETR.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger("detectron2.data.dataset_mapper").info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format. Contains height and width already
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format) # doesn't / 255
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5: # horizontal flip + resize
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else: # horizontal flip + cropping
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # not used during eval, since evaluator will load GTs
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:

                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                # this clips box to image size, hence can't be used for crowd human
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations") # pop returns
                if obj.get("iscrowd", 0) == 0
            ]
            # output box is XYXY_ABS
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

def make_transforms(cfg, is_train):
    normalize = CH_T.Compose([
        CH_T.ToTensor(), # this func will /255
        CH_T.Normalize()
    ])

    if is_train:
        scales = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        crop_size = cfg.INPUT.CROP.SIZE
        # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        return CH_T.Compose([
            CH_T.RandomHorizontalFlip(),
            CH_T.RandomSelect(
                CH_T.RandomResize(scales, max_size=max_size),
                CH_T.Compose([
                    CH_T.RandomResize([400, 500, 600]),
                    CH_T.RandomCropCH(crop_size[0], crop_size[1]),
                    CH_T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize
        ])

    if not is_train:
        scales = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        return CH_T.Compose([
            CH_T.RandomResize([scales], max_size=max_size),
            normalize
        ])


    raise ValueError(f'unknown {is_train}')

class CH_DqrfDatasetMapper:
    """

    A callable which takes a datasets dict in Detectron2 Dataset format,
    and map it into a format used by DETR.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):

        self._transform = make_transforms(cfg, is_train)
        self.image_read = FileSystemPILReader()

        logging.getLogger("detectron2.data.dataset_mapper").info(
            "Full TransformGens used in training: {}".format(str(self._transform))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    @staticmethod
    def _fake_zero_data(*size):
        return torch.zeros(size)
    @staticmethod
    def get_image_size(img):
        """ return image size in (h, w) format
        """
        w, h = img.size
        return h, w
    def __call__(self, dataset_dict):
        """
        DO NOT CLAMP FULLBOX
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # avoid modify our original list[dict]

        dataset_dict = ImageMeta.decode(dataset_dict)
        filename = dataset_dict["file_name"]

        gt_bboxes = []
        gt_bboxes_v = []
        ig_bboxes = []
        classes = []
        areas = []
        # bbox_mode = []

        for instance in dataset_dict['annotations']:  # data[-1] is instances
            if not instance['is_ignored']:  # instance[-1] is `is_ignored`
                assert instance['category_id'] > 0, '{} has invalid label {}'.format(
                    filename, instance['category_id'])
                gt_bboxes.append(instance['bbox'] )
                gt_bboxes_v.append(instance['vbbox'] )
                classes.append(instance['category_id'])
                areas.append(instance['area'])
            else:
                ig_bboxes.append(instance['bbox'])
            # bbox_mode.append(instance['bbox_mode'])

        if len(ig_bboxes) == 0:
            ig_bboxes = self._fake_zero_data(1, 4)
        if len(gt_bboxes) == 0:
            gt_bboxes = self._fake_zero_data(1, 4)
        if len(gt_bboxes_v) == 0:
            gt_bboxes_v = self._fake_zero_data(1, 4)

        image = self.image_read(filename)
        image_shape = self.get_image_size(image) # h, w
        dataset_dict["height"] = image_shape[0]
        dataset_dict["width"] = image_shape[1]

        boxes = torch.as_tensor(gt_bboxes, dtype=torch.float32).reshape(-1, 4)
        vboxes = torch.as_tensor(gt_bboxes_v, dtype=torch.float32).reshape(-1, 4)
        vboxes[:, 0::2].clamp_(min=0, max=image_shape[1])
        vboxes[:, 1::2].clamp_(min=0, max=image_shape[0])
        ig_bboxes = torch.as_tensor(ig_bboxes, dtype=torch.float32)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # bbox_mode = torch.as_tensor(bbox_mode, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0]) & (vboxes[:, 3] > vboxes[:, 1]) & (
                    vboxes[:, 2] > vboxes[:, 0])
        boxes = boxes[keep]
        vboxes = vboxes[keep]
        classes = classes[keep]
        areas = areas[keep]

        target = {"boxes": boxes, "vboxes": vboxes,
                  "iboxes": ig_bboxes, "labels": classes,
                  'area': areas}
        image, target = self._transform(image, target)
        dataset_dict["image"] = image


        if not self.is_train:
            # not used during eval, since evaluator will load GTs
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            dataset_dict.pop("annotations", None)
            dataset_dict["instances"] = target

        return dataset_dict