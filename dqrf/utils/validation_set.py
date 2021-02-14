#
# Modified by Matthieu Lin
# Contact: linmatthieu@gmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import time
import logging
import datetime
import numpy as np
import itertools
from torch.utils.data.sampler import Sampler
from torch.utils.data import DistributedSampler
from dqrf.utils.dataset_mapper import DqrfDatasetMapper, CH_DqrfDatasetMapper

from detectron2.utils.events import get_event_storage
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data.build import get_detection_dataset_dicts, trivial_batch_collator
from detectron2.engine import HookBase
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.utils.comm as comm


class ValidationLoss(HookBase): # faster than ValidationLoss
    def __init__(self, cfg, model, data_loader, weight_dict):
        super().__init__()
        self._cfg = cfg.clone()
        self._model = model
        self._period = cfg.TEST.EVAL_PERIOD
        self._data_loader = data_loader
        self._weight_dict = weight_dict
        self._data_iter = None
        self._logger_name = "detectron2.engine.hooks"
        self._logger = logging.getLogger(self._logger_name)

    def _do_eval_loss(self):
        total = len(self._data_loader)
        with torch.no_grad():
            for idx, inputs in enumerate(self._data_loader):
                loss_dict = self._model(inputs)
                device = next(iter(loss_dict.values())).device
                with torch.cuda.stream(torch.cuda.Stream() if device.type == "cuda" else None):
                    metrics_dict = {'val_' +k: v.detach().cpu().item() for k, v in loss_dict.items()}
                    all_metrics_dict = comm.gather(metrics_dict)

                if comm.is_main_process():
                    metrics_dict = {
                         k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
                    }
                    total_losses_reduced = sum(metrics_dict[k] * self._weight_dict[k.split('val_')[-1]] for k in metrics_dict.keys() if k.split('val_')[-1] in self._weight_dict)
                    if not np.isfinite(total_losses_reduced):
                        raise FloatingPointError(
                            f"Loss became infinite or NaN at iteration={idx}!\n"
                            f"loss_dict = {metrics_dict}"
                        )
                    if torch.cuda.is_available():
                        max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                    else:
                        max_mem_mb = None

                    storage = get_event_storage()
                    if len(metrics_dict) > 1:
                        storage.put_scalars(val_loss_total=total_losses_reduced,
                                            **metrics_dict)
                    log_every_n_seconds(logging.INFO,
                                        msg=" iter: {iter}/{total}  {losses}  {memory}".format(
                                            iter=idx + 1,
                                            total=total,
                                            losses="  ".join(
                                                [
                                                    "{}: {:.3f}".format(k.split('val_loss_')[-1], v.median(20))
                                                    for k, v in storage.histories().items()
                                                    if "val" in k

                                                ]
                                            ),
                                            memory="max_mem: {:.0f}M".format(
                                                max_mem_mb) if max_mem_mb is not None else ""

                                        ),
                                        n=5,
                                        name=self._logger_name
                                        )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._logger.info("Start Computing Validation Loss")
            self._do_eval_loss()

logger = logging.getLogger("detectron2.engine.hooks")
def build_detection_val_loader(cfg, dataset_name, total_batch_size,  mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """

    world_size = comm.get_world_size()
    assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    # sampler = InferenceSampler(len(dataset))
    sampler = DistributedSampler(dataset, shuffle=False)

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

class TestDistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset, but won't align the total data
    size to be divisible by world_size bacause this will lead to duplicate detecton results
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        """
        Arguments:
             - dataset (:obj:`dataset`): instance of dataset object
        """
        if num_replicas is None:
            num_replicas = comm.get_world_size()
        if rank is None:
            rank = comm.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(range(rank, len(self.dataset), num_replicas))
        self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        # g = torch.Generator()
        # g.manual_seed(self.epoch)
        # indices = list(torch.randperm(len(self.dataset), generator=g))
        indices = torch.arange(len(self.dataset))

        # subsample
        indices = indices[self.rank::self.num_replicas]
        # offset = self.num_samples * self.rank
        # indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch