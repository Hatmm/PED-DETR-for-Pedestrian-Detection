# ------------------------------------------------------------------------
# Modified by Matthieu Lin & Li Chuming
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import json
import numpy as np
from collections import Counter, OrderedDict
from dqrf.utils.box_ops import calIof, calIoU
import os
import copy
from scipy import interpolate
import bisect
import pandas

def get_scale_factor(scale, max_size, img_h, img_w):
    """
    :param scale: min size during test
    :param max_size: max size during test
    :param img_h: orig height of img
    :param img_w: orig width
    :return: scale factor for resizing
    """
    short = min(img_w, img_h)
    large = max(img_w, img_h)
    if short <= 0:
        scale_factor = 1.0
        return scale_factor
    if scale <= 0:
        scale_factor = 1.0
    else:
        scale_factor = min(scale / short, max_size / large)
    return scale_factor

class mMR_FPPI_evaluator(object):
    def __init__(self, cfg, json_file, gt_dir, watch_scale=None, class_names=None,num_classes=2, fppi=np.array([0.01, 0.1, 1]), ign_iou_thresh=None, iou_thresh=None, score_thresh=0.1, gt_rescale=True):
        """
        Follows evaluatin metric of Caltech for full body evaluation
        more informations can be found here https://github.com/Mycenae/PaperWeekly/blob/master/CrowdHuman.md
        :param json_file: path to annotation files
        :param watch_scale: scale to print
        :param class_names: class names
        :param num_classes: number of classes
        :param fppi: fppi to prints (we eval on 9 log scales evenly spaced)
        :param ign_iou_thresh: discard dts overlapping ignore region
        :param iou_thresh: threshold for true positive
        :param score_thresh: discard all dts under this score threshold
        """
        if class_names is None:
            class_names = ["background", "person"]
        if watch_scale is None:
            watch_scale = [4, 16, 32, 64, 128, 256, 512, 1024]
        if ign_iou_thresh is None:
            ign_iou_thresh = [0.5]
        if iou_thresh is None:
            iou_thresh = [0.5]
        self.min_size = cfg.INPUT.MIN_SIZE_TEST
        self.max_size = cfg.INPUT.MAX_SIZE_TEST
        self.gt_rescale = gt_rescale
        self.iou_thresh = iou_thresh
        self.ign_iou_thresh = ign_iou_thresh
        self.score_thresh = score_thresh
        self.fppi = fppi
        self.num_classes = num_classes
        self.watch_scale = watch_scale
        self.class_names = class_names
        self.gt_dir = gt_dir
        self._load_gts(json_file)


    def _load_gts(self, json_file):
        """
        :param json_file:
        :return: gts a dict containing for each label a dict that maps gts to a filename
        self.gts = {
        label: {
            filename: {
                'gts': [ "bbox": [], "vbbox": [], "detected":False ]
                }
            }
        }
        """
        self.gts = {
            'bbox_num': Counter(),
            'gt_num': Counter(),
            'image_num': 0,
            'image_scale': {},
            'gt_num_by_necessity': {}
        }
        with open(json_file, 'r') as f:
            for i, line in enumerate(f):
                json_dict = json.loads(line)
                # filename = json_dict['image_id']
                # try:
                filename = json_dict['filename']
                # except Exception:
                #     print(json_dict)

                if filename.startswith('/'):
                    filename = os.path.normpath(self.gt_dir + filename)
                else:
                    filename = os.path.join(self.gt_dir, filename)

                image_height = json_dict['image_height']
                image_width = json_dict['image_width']
                self.gts['image_scale'][filename] = get_scale_factor(self.min_size, self.max_size,image_height, image_width) if self.gt_rescale else 1.0
                self.gts['image_num'] += 1

                for instance in json_dict.get('instances', []): # for each instance in that img
                    instance['detected'] = False
                    label = instance.get('label', None)
                    # create a new reference (box_by_label) to a  gts[label] and set it equal to {}
                    box_by_label = self.gts.setdefault(label, {})
                    box_by_img = box_by_label.setdefault(filename, {'gts': []})
                    gt_by_img = box_by_img['gts']
                    gt_num_by_necessity_by_label = self.gts['gt_num_by_necessity'].setdefault(label, {})
                    self.gts['bbox_num'][label] += 1

                    #we don't evaluate on GTs with their center out of the pic as DETR's predictiong
                    #uses a sigmoid function
                    x1, y1, x2, y2 = instance['bbox']
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    cignore = cx < 0 or cx > image_width or cy < 0 or cy > image_height

                    # if not instance.get('is_ignored', False):
                    if not (instance.get('is_ignored', False) or cignore):
                        gt_by_img.append(instance)
                        self.gts['gt_num'][label] += 1
                        necessity = instance.get('necessity', None)
                        if necessity is not None:
                            gt_num_by_necessity_by_label[necessity] = gt_num_by_necessity_by_label.get(necessity, 0) + 1
                    else: # we don't evaluate pred that overlaps with ignore regions
                        ign_by_img = box_by_img.setdefault('ignores', [])
                        ign_by_img.append(instance)


    def _load_dts_list(self, res_file):
        """
        :param res_file:
        :return: a dict dts_by_label_list = {
        label: list of json annotations in COCO format
        }
        """
        dts_by_label_list = {}
        with open(res_file, 'r') as f:
            for line in f:
                # try:
                dt = json.loads(line)
                # except Exception:
                #     print("################ LINE ################")
                #     print(line)
                #     print("################ LINE ################")
                if dt['score'] < self.score_thresh:
                    continue
                dt_by_label = dts_by_label_list.setdefault(dt['label'], []) # create new dict dt['label']: []
                dt_by_label.append(dt)

        return dts_by_label_list

    def eval(self, dt_file):
        return_dict = {}
        dts_by_label_list = self._load_dts_list(dt_file)

        total_ap = np.zeros([len(self.iou_thresh)])
        total_mmr = np.zeros([len(self.iou_thresh)])
        total_ar = np.zeros([len(self.iou_thresh)])
        total_mr = np.zeros([len(self.iou_thresh), len(self.fppi)])

        for iou_i, (iou_thresh, ign_iou_thresh) in enumerate(zip(self.iou_thresh, self.ign_iou_thresh)):
            self._reset_detected_flag()
            #metrics for each class
            ap = np.zeros(self.num_classes)
            mmr = np.zeros(self.num_classes)
            max_recall = np.zeros(self.num_classes)
            fppi_miss = np.zeros([self.num_classes, len(self.fppi)])
            fppi_scores = np.zeros([self.num_classes, len(self.fppi)])
            for class_i in range(1, self.num_classes): # background class is set to be 0
                sum_box = self.gts['bbox_num'][class_i]
                sum_gt = self.gts['gt_num'][class_i]
                gt_num_by_necessity = self.gts['gt_num_by_necessity'][class_i]
                print(gt_num_by_necessity)
                results_i = dts_by_label_list.get(class_i, [])
                print(f'sum_gt vs sum_dt: {sum_gt} vs {len(results_i)}')
                if sum_box == 0 or len(results_i) == 0:
                    continue
                tp, fp, matched_gt_minsize, matched_gt_necessities = self._get_cls_tp_fp(
                    results_i, self.gts[class_i], self.gts['image_scale'], iou_thresh, ign_iou_thresh
                )
                drec = tp / max(1, sum_gt) # tp / nb_gt
                tp = np.cumsum(tp) #[#nb of tp, ]
                fp = np.cumsum(fp) #[#nb of fp, ]
                rec = tp / sum_gt # tp / nb_gt
                prec = tp / np.maximum(tp + fp, 1)
                for v in range(len(prec) - 2, -1, -1): # interpolate
                    prec[v] = max(prec[v], prec[v + 1])
                scores = sorted([x['score'] for x in results_i], reverse=True)
                mrs, s_fppi = self._get_miss_rate(tp, fp, scores, self.gts['image_num'], sum_gt)


                if len(self.watch_scale) > 0:
                    multi_size_metrics = self.get_miss_rate_multi_size(
                        tp, fp, scores, self.gts, class_i, matched_gt_minsize, self.watch_scale)
                    for k, v in multi_size_metrics.items():
                        print("{}:\t{}*{}".format(self.class_names[class_i], k, v))

                if len(gt_num_by_necessity) > 0:
                    multi_necessity_metrics = self.get_miss_rate_multi_necessity(
                        tp, fp, self.gts, class_i, matched_gt_necessities)
                    for k, v in multi_necessity_metrics.items():
                        print("{}:\t{}: {}".format(self.class_names[class_i], k, v))

                ap[class_i] = np.sum(drec * prec)
                mmr[class_i] = self.get_cls_mmr(tp, fp, self.gts['image_num'], sum_gt)
                max_recall[class_i] = np.max(rec)
                fppi_miss[class_i] = 1 - mrs
                fppi_scores[class_i] = s_fppi
                return_dict[f'iou{iou_thresh}-{self.class_names[class_i]}-mAP'] = ap[class_i]
                return_dict[f'iou{iou_thresh}-{self.class_names[class_i]}-mMR'] = mmr[class_i]
                return_dict[f'iou{iou_thresh}-{self.class_names[class_i]}-max_recall'] = max_recall[class_i]
                for fppi_i, fppi in enumerate(self.fppi):
                    return_dict[f'iou{iou_thresh}-{self.class_names[class_i]}-fppi{self.fppi[fppi_i]}'] = fppi_miss[class_i][fppi_i]

                # import pdb
                # pdb.set_trace()

            mAP = np.mean(ap[ap.nonzero()])
            mmmr = np.mean(mmr[mmr.nonzero()])
            m_rec = np.mean(max_recall[max_recall.nonzero()])
            m_fppi_miss = []
            for fppi_i, fppi in enumerate(self.fppi):
                nzero = fppi_miss[:, fppi_i][fppi_miss[:, fppi_i].nonzero()]
                if len(nzero) > 0:
                    m_fppi_miss.append(np.mean(nzero))
                else:
                    m_fppi_miss.append(0)
            m_fppi_miss = np.array(m_fppi_miss)

            print("\n\n-----------------------------------------------------------------------------------")
            print("\t* fppi val = {} iou = {}".format(self.fppi, iou_thresh))
            print("\t--------------------------------------------------------------------------")
            print("\tClass\t|AP\t|mMR\t|Recall\t|fppi-miss\t\t |fppi-score")
            print("\t--------------------------------------------------------------------------")
            for i in range(1, self.num_classes):
                s = "\t{}:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(self.class_names[i], ap[i] * 100, mmr[i] * 100,
                                                                   max_recall[i] * 100,
                                                                   fppi_miss[i][0] * 100)
                print(s)
                s_detail = "\t{}:\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{}|{}".format(self.class_names[i], ap[i], mmr[i],
                                                                             max_recall[i],
                                                                             self._np_fmt(fppi_miss[i]),
                                                                             self._np_fmt(fppi_scores[i]))
                print(s_detail)
                print("\t--------------------------------------------------------------------------")
            np.set_printoptions(precision=4)
            print(
                '\n\tmAP: {:.4f}   mmMR: {:.4f}   max recall: {:.4f} mean fppi-mr: {}'.format(mAP, mmmr, m_rec,
                                                                                                 m_fppi_miss))
            print("-----------------------------------------------------------------------------------\n")

            total_ap[iou_i] = mAP
            total_mmr[iou_i] = mmmr
            total_ar[iou_i] = m_rec
            total_mr[iou_i] = m_fppi_miss

        for iou_i, (iou_thresh, ign_iou_thresh) in enumerate(zip(self.iou_thresh, self.ign_iou_thresh)):
            return_dict[f'total-iou{iou_thresh}-mAP'] = total_ap[iou_i]
            return_dict[f'total-iou{iou_thresh}-mmMR'] = total_mmr[iou_i]
            return_dict[f'total-iou{iou_thresh}-max_recall'] = total_ar[iou_i]
            for fppi_i, fppi in enumerate(self.fppi):
                return_dict[f'total-iou{iou_thresh}-fppi{self.fppi[fppi_i]}'] = total_mr[iou_i][fppi_i]
        for k,v in return_dict.items():
            return_dict[k] = round(v, 4)

        return return_dict

    def _np_fmt(self, x):
        s = "["
        for i in list(x):
            s += " {:.4f}".format(i)
        s += " ]"
        return s
    def get_cls_mmr(self, tp, fp, image_num, gt_num):
        """ Computes log average miss-rate from a MR-FPPI-curve.

        The log average miss-rate is defined as the average of a number of evenly spaced log miss-rate samples
        on the :math:`{log}(FPPI)` axis within the range :math:`[10^{-2}, 10^{0}]`

        Returns:
            Number: log average miss-rate
        """

        m = 1 - tp / gt_num
        f = fp / image_num
        if f[-1] == 0:
            return m[-1]
        s = np.logspace(-2., 0., 9)
        interpolated = interpolate.interp1d(f, m, fill_value=(1., 0.), bounds_error=False)(s)
        log_interpolated = np.log(interpolated)

        avg = sum(log_interpolated) / len(log_interpolated)
        return np.exp(avg)
    def get_miss_rate_multi_size(self, tp, fp, scores, gts_list_i, class_i, matched_gt_minsize, watch_scale):
        """
        input: accumulated tps & fps
        len(tp) == len(fp) ==len(scores) == len(box)
        """
        image_num = gts_list_i['image_num']
        gt_num = gts_list_i['gt_num'][class_i]
        gt_class_i = gts_list_i[class_i]
        image_scale = gts_list_i['image_scale']
        N = len(self.fppi)
        maxfps = self.fppi * image_num
        mrs = np.zeros(N)
        fppi_scores = np.zeros(N)

        multi_size_metrics = {}
        gt_bboxes = []
        gt_scales = []
        for img_id in gt_class_i:
            gts = gt_class_i[img_id]['gts']
            gt_bboxes.extend([g['bbox'] for g in gts])
            gt_scales.extend([image_scale[img_id] for g in gts])
        gt_bboxes = np.stack(gt_bboxes)
        gt_scales = np.stack(gt_scales)
        box_w = gt_bboxes[:,2] - gt_bboxes[:,0]
        box_h = gt_bboxes[:,3] - gt_bboxes[:,1]
        gt_minsize = np.minimum(box_w, box_h)
        gt_minsize = gt_minsize * gt_scales
        assert gt_num == gt_minsize.shape[0], f'gt_num vs gt_minsize: {gt_num} vs {gt_minsize.shape}'
        gt_hist, edges = np.histogram(gt_minsize, bins=watch_scale)

        for i, f in enumerate(maxfps):
            idxs = np.where(fp > f)[0]
            if len(idxs) > 0:
                idx = idxs[0]  # the last fp@fppi
            else:
                idx = -1  # no fps, tp[-1]==gt_num
            mrs[i] = 1 - tp[idx] / gt_num
            fppi_scores[i] = scores[idx]

            this_fppi_matched_gt_minsize = copy.deepcopy(matched_gt_minsize)
            this_fppi_matched_gt_minsize = this_fppi_matched_gt_minsize[:idx]
            this_fppi_matched_gt_minsize = this_fppi_matched_gt_minsize[this_fppi_matched_gt_minsize>0]
            matched_gt_hist, edges = np.histogram(this_fppi_matched_gt_minsize, bins=watch_scale)

            for bin_i in range(gt_hist.shape[0]):
                percent = gt_hist[bin_i] * 1.0 / max(1.0, gt_num)
                recall = matched_gt_hist[bin_i] * 1.0 / (gt_hist[bin_i] + 0.000001)
                miss_rate = 1 - recall
                base_fp_head = f'fppi-{self.fppi[i]}-'
                base_fp_size_head = base_fp_head + 'size {:4d}-{:4d}-recall_percent {:.4f}'.format(
                    watch_scale[bin_i], watch_scale[bin_i + 1], percent)
                multi_size_metrics[base_fp_size_head] = round(recall, 4)
        return multi_size_metrics
    def get_miss_rate_multi_necessity(self, tp, fp, gts_list_i, class_i, matched_gt_necessities):
        """
        input: accumulated tps & fps
        len(tp) == len(fp) ==len(scores) == len(box)
        """
        image_num = gts_list_i['image_num']
        gt_num_by_necessity = gts_list_i['gt_num_by_necessity'][class_i]
        maxfps = self.fppi * image_num
        multi_necessity_metrics = {}

        for i, f in enumerate(maxfps):
            idxs = np.where(fp > f)[0]
            if len(idxs) > 0:
                idx = idxs[0]  # the last fp@fppi
            else:
                idx = -1  # no fps, tp[-1]==gt_num
            for k,v in gt_num_by_necessity.items():
                recall = sum([i==k for i in matched_gt_necessities[:idx]]) * 1.0 / v
                base_fp_head = f'fppi-{self.fppi[i]}-'
                base_fp_necessity_head = base_fp_head + 'necessity {} num {}'.format(k, v)
                multi_necessity_metrics[base_fp_necessity_head] = round(recall, 4)

        return multi_necessity_metrics
    def _get_miss_rate(self, tp, fp, scores, image_num, gt_num, return_index=False):
        """
        input: accumulated tps & fps
        len(tp) == len(fp) == len(scores) == len(box)
        """
        N = len(self.fppi)
        maxfps = self.fppi * image_num
        mrs = np.zeros(N)
        fppi_scores = np.zeros(N)

        indices = []
        for i, f in enumerate(maxfps):
            idxs = np.where(fp > f)[0]
            if len(idxs) > 0:
                idx = idxs[0]  # the last fp@fppi
            else:
                idx = -1  # no fps, tp[-1]==gt_num
            indices.append(idx)
            mrs[i] = 1 - tp[idx] / gt_num
            fppi_scores[i] = scores[idx]
        if return_index:
            return mrs, fppi_scores, indices
        else:
            return mrs, fppi_scores
    def _get_cls_tp_fp(self, dts_cls, gts_cls, image_scale, iou_tresh, ign_iou_thresh):
        """
        :param dts_cls: results for a given class
        :param gts_cls: gts for a given class
        :param image_scale: scale to original size
        :param iou_tresh: iou threshold for tp
        :param ign_iou_thresh: iou threshold for ignored dt
        :return: sorted binary array for tps, fps from highest score to smallest
        """
        fps, tps = np.zeros((len(dts_cls))), np.zeros((len(dts_cls))) # [#dts]
        necessities = []
        matched_gt_minsize = np.ones((len(dts_cls))) * -1 # minsize for each gts
        dts_cls = sorted(dts_cls, key=lambda x: -x['score']) # sort from highest score to smallest
        for i, dt in enumerate(dts_cls):
            img_id = dt['image_id']
            dt_bbox = dt['bbox'] # 1
            m_iou, m_gt, m_iof = -1, -1, -1
            if img_id in gts_cls:
                gts = gts_cls[img_id]
                gt_bboxes = [g['bbox'] for g in gts['gts']] # M
                ign_bboxes = [g['bbox'] for g in gts['ignores']] if 'ignores' in gts else []
                m_iou, m_gt, m_iof = self.match(dt_bbox, gt_bboxes, ign_bboxes) #[1, 1]
            if m_iou >= iou_tresh: # iou dec is with matched gt is higher than thresh its a true positive
                if not gts['gts'][m_gt]['detected']: #gt not detected yet
                    tps[i] = 1
                    fps[i] = 0
                    necessities.append(gts['gts'][m_gt].get('necessity','none'))
                    gts['gts'][m_gt]['detected'] = True
                    m_gt_bboxes = gt_bboxes[m_gt]
                    box_w = m_gt_bboxes[2] - m_gt_bboxes[0]
                    box_h = m_gt_bboxes[3] - m_gt_bboxes[1]
                    matched_gt_minsize[i] = np.minimum(box_w, box_h)
                    matched_gt_minsize[i] *= image_scale[img_id]
                else:
                    fps[i] = 1
                    tps[i] = 0
                    necessities.append('fp')
            else:
                fps[i] = 1
                tps[i] = 0
                necessities.append('fp')

            if fps[i] == 1 and m_iof >= ign_iou_thresh:
                fps[i] = 0
                tps[i] = 0
                necessities.append('ig')
        return np.array(tps), np.array(fps), np.array(matched_gt_minsize), necessities

    def match(self, dt, gts, igns):
        dt = np.array(dt).reshape(-1, 4)
        gts = np.array(gts)
        if len(gts) > 0:
            ious = calIoU(dt, gts).reshape(-1)
            matched_gt = ious.argmax() # matches as COCO, highest IoU if multiple DT matches to one GT
            matched_iou = ious[matched_gt]
        else:
            matched_iou = -1
            matched_gt = -1
        if len(igns) > 0:
            igns = np.array(igns)
            iofs = calIof(dt, igns).reshape(-1)
            maxiof = np.max(iofs)
        else:
            maxiof = 0
        return matched_iou, matched_gt, maxiof

    def _reset_detected_flag(self):
        """
        Set all instance's detected flag to False
        """
        for cls in range(1, self.num_classes):
            if cls in self.gts.keys():
                for img_id, gts in self.gts[cls].items():
                    for instance in gts['gts']:
                        instance['detected'] = False









