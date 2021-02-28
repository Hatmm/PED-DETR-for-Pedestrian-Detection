#
# Modified by Matthieu Lin
# Contact: linmatthieu@gmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.engine.hooks import HookBase
import datetime
import time
import torch
import logging



class TrainingMetricPrinter(EventWriter):
    """
    Similar to CommongMetricPrinter but only prints training loss
    """

    def __init__(self, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self.logger = logging.getLogger("detectron2.utils.events")
        self._max_iter = max_iter
        self._last_write = None

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None

        eta_string = None
        try:
            iter_time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None
            # estimate eta on our own - more noisy
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                        iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())

        try:
            lr = "{:.6f}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            " {eta}iter: {iter}/{total_iter}  {losses}  {time}{data_time}lr: {lr}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                total_iter=self._max_iter,
                losses="  ".join(
                    [
                        "{}: {:.3f}".format(k.split('train_loss_')[-1], v.median(20))
                        for k, v in storage.histories().items()
                        if ("train" in k and "AP" not in k)
                    ]
                ),
                time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )


class InitMetricPrinter(EventWriter):
    """
    Similar to CommongMetricPrinter but only prints training loss
    """
    logger = logging.getLogger("detectron2.utils.events")
    def __init__(self):
        super(InitMetricPrinter, self).__init__()

    @classmethod
    def write(cls):
        storage = get_event_storage()
        iteration = storage.iter

        try:
            lr = "{:.6f}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        cls.logger.info(
            "iter: Initialization values  {losses}  lr: {lr}  {memory}".format(
                iter=iteration,
                losses="  ".join(
                    [
                        "{}: {:.3f}".format(k.split('train_loss_')[-1], v.latest())
                        for k, v in storage.histories().items()
                        if ("train" in k and "AP" not in k)
                    ]
                ),
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )


class PeriodicWriter_withInitLoss(HookBase):
    """
    same as PeriodicWriter but print initial loss which is useful for debugging loss
    Write events to EventStorage (by calling ``writer.write()``) periodically.

    It is executed every ``period`` iterations and after the last iteration.
    Note that ``period`` does not affect how data is smoothed by each writer.
    """

    def __init__(self, writers, period=20):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0 or (
            self.trainer.iter == self.trainer.max_iter - 1
        ):
            for writer in self._writers:
                writer.write()
        elif (self.trainer.iter +1) == 1:
            InitMetricPrinter.write()
            # for writer in self._writers:
            #     if isinstance(writer, InitMetricPrinter):
            #         writer.write()

    def after_train(self):
        for writer in self._writers:
            # If any new data is found (e.g. produced by other after_train),
            # write them before closing
            writer.write()
            writer.close()