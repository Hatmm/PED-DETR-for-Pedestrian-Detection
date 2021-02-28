# Author: Li Chuming
# contact lichuming@sensetime.com
import argparse
import json
import os

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="extract frame from video list")
    parser.add_argument("-f", type=str)

    parser.add_argument("--input_file", type=str, default="", help="")
    parser.add_argument("--output_file", type=str, default="", help="output json")

    args = parser.parse_args()
    return args


args = parse_args()

args.image_root = 'path/val2017/'
args.input_file = 'path/annotation_val.odgt'
args.output_file = 'CrowdHuman_val.json'


PERSON_LABEL = 1


def Convert2POD(args):
    f = open(args.input_file)
    output_list = []
    image_count = 0
    box_count = 0
    ignore_count = 0
    for line in tqdm(f):
        """
        {
        "ID": "273271,1c72c000a2ee47d5",
        "gtboxes": [
        {"fbox": [23, 230, 228, 597], "tag": "person", "hbox": [125, 233, 82, 110], "extra": {"box_id": 0, "occ": 1}, "vbox": [62, 234, 186, 487], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}},
        {"fbox": [0, 308, 164, 416], "tag": "mask", "hbox": [0, 308, 164, 416], "extra": {"ignore": 1}, "vbox": [0, 308, 164, 416], "head_attr": {}}
        ]
        }
        """
        line = json.loads(line.strip())
        json_data = {}
        json_data["filename"] = line['ID'] + '.jpg'
        img = cv2.imread(os.path.join(args.image_root, json_data["filename"]))
        json_data["image_height"] = img.shape[0]
        json_data["image_width"] = img.shape[1]
        json_data["instances"] = []
        for gtbox in line['gtboxes']:
            fbox = gtbox['fbox']
            tag = gtbox['tag']
            hbox = gtbox['hbox']
            extra = gtbox['extra']
            vbox = gtbox['vbox']
            head_attr = gtbox['head_attr']

            is_ignored = extra.get("ignore", 0)
            is_ignored = {0: False, 1: True}[is_ignored]
            if is_ignored:
                ignore_count += 1
                assert tag in {'mask', 'person'}, '{} {}'.format(line['ID'], gtbox)
            else:
                box_count += 1
                assert tag == 'person', tag
            label = PERSON_LABEL
            bbox = fbox
            vbbox = vbox
            bbox[2] += bbox[0] - 1
            bbox[3] += bbox[1] - 1
            vbbox[2] += vbbox[0] - 1
            vbbox[3] += vbbox[1] - 1
            json_data["instances"].append(
                {
                    "is_ignored": is_ignored,
                    "bbox": bbox,
                    "vbbox": vbbox,
                    "label": label,
                }
            )
        image_count += 1
        output_list.append(json_data)

    print('total {} images, {} boxes, {} ignores'.format(image_count, box_count, ignore_count))

    def write_json_file(output_list, filename):
        f = open(filename, 'w')
        print('start to write file {}'.format(filename))
        for line in tqdm(output_list):
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
        f.close()

    write_json_file(output_list, args.output_file)


Convert2POD(args)