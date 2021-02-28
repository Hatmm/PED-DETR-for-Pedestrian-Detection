#
# Modified by Matthieu Lin
# Contact: linmatthieu@gmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import contextlib
import os
import json
import logging
from collections import OrderedDict
import pickle
import copy
import numpy as np
import torch
from fvcore.common.file_io import PathManager
import itertools
from collections import Counter

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from detectron2.evaluation import DatasetEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from dqrf.utils.uni_evaluator import mMR_FPPI_evaluator


class CrowdHumanEvaluator(DatasetEvaluator):
    """
    Evaluate mMR and AP for CrowdHuman
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(self,
                 dataset_name,
                 cfg,
                 distributed,
                 output_dir=None):

        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger("detectron2.evaluation.evaluator")

        # metadatacatalog return for a given dataset name its metadata e.g its attribute of the class
        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.info(
                f"'json file was not found in MetaDataCatalog for {dataset_name}'")

        self.mMR_FPPI_evaluator = mMR_FPPI_evaluator(cfg, self._metadata.json_file, gt_dir=self._metadata.gt_dir, gt_rescale=True)
    def _tasks_from_config(self, cfg):
        """
                Returns:
                    tuple[str]: tasks that can be evaluated under the given configuration.
                """
        tasks = ("bbox",)
        # if cfg.MODEL.MASK_ON:
        #     tasks = tasks + ("segm",)
        # if cfg.MODEL.KEYPOINT_ON:
        #     tasks = tasks + ("keypoints",)
        return tasks
    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predictions = []



    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input['image_id']}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                """
                When training in distributed dataset size must be divisible by the world size
                or else it will duplicate data such that the total data size is divisble
                in order to remove duplicate predictions coco uses np.unique(), here we avoid duplicate
                predictions by checking if it already been processed by checking the file_names
                """
                # coco converter xyxy -> xywh
                prediction["instances"] = instances_to_crowdhuman_json(
                    instances, input["file_name"] # pair prediction with file_name instead of img_id
                )
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(
                    self._cpu_device)


            self._predictions.append(prediction)

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0) #a list of list for each rank
            predictions = list(itertools.chain(*predictions)) #concat the list

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[CrowdHuman Evaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)
        # Copy such that we can modify it in trainer if needed
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for CrowdHuman format ...")
        crowdhuman_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "crowdhuman_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                for line in crowdhuman_results:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                self.mMR_FPPI_evaluator.eval(file_path)
                # _evaluate_predictions_on_crowdhuman(
                #     self._metadata.json_file, file_path
                # )
            if len(crowdhuman_results) > 0
            else None # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, self._metadata.thing_classes
            )
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or List): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score} used by print_csv_format to print and saved in trainer storage
        """
        results = {}
        # metrics = ["AP", "mMR", "Recall"] # add FPPI etc
        metrics = {
            "bbox": ["mAP", "mMR", "max_recall", "fppi0.01", "fppi0.1", "fppi1.0"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warning("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            k: v for k, v in coco_eval.items() if 'total' in k
        }
        # results = {metric: coco_eval[idx]
        #            for idx, metric in enumerate(metrics)}
        small_table = create_small_table(results)
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + small_table
        )
        # print(coco_eval)

        return results


def instances_to_crowdhuman_json(instances, filename):
    """
    :param instances: pred_boxes [num_query, 4], pred_classes [num_query],
    :param filename:
    :return:
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes#.tensor#.tolist()
    scores = instances.scores#.tolist()
    classes = instances.pred_classes#.tolist()

    results = []
    for box_score, cls, box in zip(scores, classes, boxes):
        if box_score > 0.1:
            xmin, ymin, xmax, ymax = round(float(box[0]), 1), round(float(box[1]), 1), round(float(box[2]), 1), round(float(box[3]), 1)
            box_score = round(box_score.item(), 5)
            cls = int(cls) # convert torch.tensor([1.]) to 1.
            result = {
                'image_id': filename,
                'bbox': [xmin, ymin, xmax, ymax],
                'score': box_score,
                'label': cls,
            }
            results.append(result)

    return results
