# ------------------------------------------------------------------------
# Modified by Matthieu Lin
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from detectron2.config import CfgNode as CN

def add_dataset_path(cfg):
    """
    Add config for dataset path
    """

    _C = cfg

    _C.CH_PATH = CN()

    _C.CH_PATH.ANNOT_PATH_TRAIN = "PATH/CrowdHuman_train.json"
    _C.CH_PATH.IMG_PATH_TRAIN = "PATH/train2017"
    _C.CH_PATH.ANNOT_PATH_VAL = "PATH/CrowdHuman_val.json"
    _C.CH_PATH.IMG_PATH_VAL = "PATH/val2017"

def add_dqrf_config(cfg):
    """
    Add config for DQRF
    """
    cfg.MODEL.DQRF_DETR  = CN()
    cfg.MODEL.DQRF_DETR.NUM_CLASSES = 80
    cfg.MODEL.DQRF_DETR.NUM_QUERIES = 300
    cfg.MODEL.DQRF_DETR.FOCAL_ALPHA = 0.25
    cfg.MODEL.DQRF_DETR.GAMMA = 2.0

    #TRANSFORMER
    cfg.MODEL.DQRF_DETR.HIDDEN_DIM = 256
    cfg.MODEL.DQRF_DETR.NHEAD = 8
    cfg.MODEL.DQRF_DETR.NUM_DECODER_LAYERS = 6
    cfg.MODEL.DQRF_DETR.NUM_ENCODER_LAYERS = 6
    cfg.MODEL.DQRF_DETR.DIM_FEEDFORWARD = 1024
    cfg.MODEL.DQRF_DETR.ACTIVATION = "relu"
    cfg.MODEL.DQRF_DETR.DROPOUT = 0.1
    cfg.MODEL.DQRF_DETR.NUM_FEATURE_LEVELS = 4
    cfg.MODEL.DQRF_DETR.ENC_SAMPLING_POINTS = 4
    cfg.MODEL.DQRF_DETR.DEC_SAMPLING_POINTS = 4
    cfg.MODEL.DQRF_DETR.POS_ENCODING = 'sine'

    #LOSS
    cfg.MODEL.DQRF_DETR.AUX_LOSS = True
    cfg.MODEL.DQRF_DETR.V_MATCH = 3
    cfg.MODEL.DQRF_DETR.COST_CLASS = 2
    cfg.MODEL.DQRF_DETR.COST_BBOX = 5
    cfg.MODEL.DQRF_DETR.COST_GIOU = 2

    #DQRF
    cfg.MODEL.DQRF_DETR.DENSE_QUERY = False
    cfg.MODEL.DQRF_DETR.RECTIFIED_ATTENTION = False
    cfg.MODEL.DQRF_DETR.IGNORE_IOU_THRESHOLD = 0.5

    #SOLVER
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.SAMPLE_MULTIPLIER = 0.1
    cfg.SOLVER.CENTER_MULTPLIER = 0.1
    cfg.SOLVER.WEIGHT_MULTIPLIER = 1.0

