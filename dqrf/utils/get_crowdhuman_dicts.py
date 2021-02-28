#
# Modified by Matthieu Lin
# Contact: linmatthieu@gmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
import contextlib
import logging
import json
import os
from dqrf.utils.utils import ImageMeta
"""
This file contains functions to parse Crowdhuman-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger("detectron2.data.datasets.coco")

class get_crowdhuman_dicts(object):
    def __init__(self, json_file, image_root):
        """
        Note that crowdhuman box format is XYXY
        :param json_file: str, full path to the json file in CH instances annotation format.
        :param image_root: str or path-like, the directory where the images in this json file exists
        :return: list[dict] each dict contains file_name, image_id, height, width,
        """
        self.json_file = json_file
        self.image_root = image_root

    def __call__(self):
        timer = Timer()
        json_file = PathManager.get_local_path(self.json_file)
        with open(json_file, 'r') as file:
            # imgs_anns = json.load(file)
            imgs_anns = file.readlines()
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

        logger.info("Loaded {} images in CrowdHuman format from {}".format(
            len(imgs_anns), json_file))

        dataset_dicts = []
        # aspect_ratios = []

        for idx, ann in enumerate(imgs_anns):
            v = json.loads(ann)
            record = {}

            filename = v["filename"]
            # NOTE when filename starts with '/', it is an absolute filename thus os.path.join doesn't work
            if filename.startswith('/'):
                filename = os.path.normpath(self.image_root + filename)
            else:
                filename = os.path.join(self.image_root, filename)
            height, width = v["image_height"], v["image_width"]

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            objs = []
            for anno in v.get('instances', []):
                x1, y1, x2, y2 = anno['bbox']
                w = x2 - x1
                h = y2 - y1
                obj = {
                    "category_id": anno['label'],
                    "bbox": anno['bbox'],
                    "vbbox": anno['vbbox'],
                    "is_ignored": anno.get('is_ignored', False),
                    'area': w * h,
                    # 'bbox_mode': BoxMode.XYXY_ABS
                }
                objs.append(obj)
            # ratio = 1.0 * (height + 1) / (width + 1) # do something with ratio ?
            record["annotations"] = objs
            # dataset_dicts.append(record) # to print class histogram
            dataset_dicts.append(ImageMeta.encode(record)) #this saves up to x2 memory when serializing the data
            # aspect_ratios.append(ratio)
        return dataset_dicts







