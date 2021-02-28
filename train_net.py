# ------------------------------------------------------------------------
# Modified by Matthieu Lin
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import contextlib
import logging
import os
import time
from typing import Any, Dict, List, Set
import numpy as np
import detectron2.utils.comm as comm
import torch
import itertools
from torch.nn.parallel import DistributedDataParallel
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetCatalog,
    MetadataCatalog)
from detectron2.data.samplers import InferenceSampler
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator, DatasetEvaluator
)
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import get_event_storage, JSONWriter, TensorboardXWriter

from dqrf import add_dqrf_config, add_dataset_path
from dqrf.utils.get_crowdhuman_dicts import get_crowdhuman_dicts
from dqrf.utils.dataset_mapper import DqrfDatasetMapper, CH_DqrfDatasetMapper
from dqrf.utils.ch_evalutor import CrowdHumanEvaluator
from dqrf.utils.validation_set import ValidationLoss, ValidationLoss_2, build_detection_val_loader
from dqrf.utils.metric_writer import TrainingMetricPrinter, PeriodicWriter_withInitLoss
try:
    _nullcontext = contextlib.nullcontext  # python 3.7+
except AttributeError:

    @contextlib.contextmanager
    def _nullcontext(enter_result=None):
        yield enter_result
logger = logging.getLogger("detectron2")


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DQRF.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.clip_norm_val = 0.0
        if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
                self.clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        super().__init__(cfg)

        # overwrite model to use new pytorch feature for balanced gradient
        # model = self.build_model(cfg)
        # if comm.get_world_size() > 1:
        #     model = DistributedDataParallel(
        #         model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
        #       gradient_as_bucket_view=True
        #     )

    def build_hooks(self):
        """
        list of default hooks in order: IterationsTimer, LRScheduler, PreciseBN (disabled here),
        Checkpointer, Evalhook, Writer
        :return: list[HookBase]
        """
        if comm.get_world_size() > 1:
            # logger.info(f'Distributed training active on {[comm.get_local_rank()]}')
            self.weight_dict = self.model.module.criterion.weight_dict
        else:
            self.weight_dict = self.model.criterion.weight_dict

        hooks = super().build_hooks()


        hooks.insert(-2, ValidationLoss_2(
            self.cfg,
            self._trainer.model,
            build_detection_val_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                self.cfg.SOLVER.IMS_PER_BATCH,
                mapper=self._return_val_mapper()),
            self.weight_dict))  # insert before writer and eval hook

        # if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            # hooks.pop()
            # we keep an average of the last 20 values e.g EMA with Beta ~0.95 since 20 = 1 / (1 - beta)
            # hooks.append(PeriodicWriter_withInitLoss(self.build_writers(), period=20))
        # hooks.insert(-2, ValidationLoss(
        #     self.cfg,
        #     self.model,
        # self.weight_dict)) #insert before writer and eval hook

        return hooks

    def build_writers(self):
        return [
            TrainingMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def _return_val_mapper(self):
        if 'coco' in self.cfg.DATASETS.TRAIN[0]:
            mapper = DqrfDatasetMapper(self.cfg, True)
        elif 'crowd' in self.cfg.DATASETS.TRAIN[0]:
            mapper = CH_DqrfDatasetMapper(self.cfg, True)
        else:
            mapper = None
        return mapper

    def _write_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        device = next(iter(loss_dict.values())).device

        # Use a new stream so these ops don't wait for DDP or backward
        with torch.cuda.stream(torch.cuda.Stream() if device.type == "cuda" else None):
            metrics_dict = {'train_' + k: v.detach().cpu().item() for k, v in loss_dict.items()}
            metrics_dict["data_time"] = data_time

            # Gather metrics among all workers for logging
            # This assumes we do DDP-style training, which is currently the only
            # supported method in detectron2.
            all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            # Note these are unscaled
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            #TODO ugly work around
            total_losses_reduced = sum(metrics_dict[k] * self.weight_dict[k.split('train_')[-1]] for k in metrics_dict.keys() if k.split('train_')[-1] in self.weight_dict)
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("train_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)


    def run_step(self):
        assert self._trainer.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._trainer._data_loader_iter) # work around for last detectron2 version
        # data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self._trainer.model(data)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        self._trainer.optimizer.zero_grad()
        losses.backward()

        # self._trainer._write_metrics(loss_dict, data_time)
        self._write_metrics(loss_dict, data_time)
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        if self.clip_norm_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self._trainer.model.parameters(), self.clip_norm_val)
        self._trainer.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'coco' in cfg.DATASETS.TEST[0]:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif 'crowd' in cfg.DATASETS.TEST[0]:
            return CrowdHumanEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if 'coco' in cfg.DATASETS.TRAIN[0]:
            mapper = DqrfDatasetMapper(cfg, False)
        elif 'crowd' in cfg.DATASETS.TRAIN[0]:
            mapper = CH_DqrfDatasetMapper(cfg, False)
        else:
            mapper = None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        if 'coco' in cfg.DATASETS.TRAIN[0]:
            mapper = DqrfDatasetMapper(cfg, True)
        elif 'crowd' in cfg.DATASETS.TRAIN[0]:
            mapper = CH_DqrfDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of parameters {n_parameters}")

        def is_backbone(n, backbone_names):
            out = False
            for b in backbone_names:
                if b in n:
                    out = True
                    break
            return out

        #careful DEFORMABLE DETR yields poorer performance is its FAKE FPN is trained on the same LR as Resnet
        #Resnet parameters name is backbone.0
        lr_backbone_names = ['backbone.0']

        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not is_backbone(n, lr_backbone_names) and
                           not (
                                       "roi_fc1" in n or "roi_fc2" in n or "offset" in n or "sampling_locs" in n or "sampling_cens" in n or "sampling_weight" in n or "conv_offset" in n or 'learnable_fc' in n) and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR,
            },
            {
                "params": [p for n, p in model.named_parameters() if is_backbone(n, lr_backbone_names) and
                           not (
                                       "roi_fc1" in n or "roi_fc2" in n or "offset" in n or "sampling_locs" in n or "sampling_cens" in n or "sampling_weight" in n or "conv_offset" in n or 'learnable_fc' in n) and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_MULTIPLIER,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           ("sampling_locs" in n) and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.SAMPLE_MULTIPLIER,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           ("sampling_cens" in n) and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.CENTER_MULTPLIER,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           ("sampling_weight" in n) and p.requires_grad],
                "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.WEIGHT_MULTIPLIER,
            },

        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.SOLVER.BASE_LR,
                                      weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        return optimizer
    
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_dqrf_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if "crowd" in cfg.DATASETS.TRAIN[0]:
        add_dataset_path(cfg)
        ch_train = get_crowdhuman_dicts(cfg.CH_PATH.ANNOT_PATH_TRAIN, cfg.CH_PATH.IMG_PATH_TRAIN)
        ch_val = get_crowdhuman_dicts(cfg.CH_PATH.ANNOT_PATH_VAL, cfg.CH_PATH.IMG_PATH_VAL)
        DatasetCatalog.register(cfg.DATASETS.TRAIN[0], ch_train)
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=["Background", "Pedestrian"])
        DatasetCatalog.register(cfg.DATASETS.TEST[0], ch_val)
        MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(thing_classes=["Background", "Pedestrian"])
        MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(json_file=cfg.CH_PATH.ANNOT_PATH_VAL)
        MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(gt_dir=cfg.CH_PATH.IMG_PATH_VAL)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
       
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)
    """
    A torch process group which only includes processes that on the same machine as the current process.
    This variable is set when processes are spawned by `launch()` in "engine/launch.py".
    """
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
