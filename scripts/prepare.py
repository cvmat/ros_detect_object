#!/usr/bin/env python
import argparse
import json
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16

from chainer_npz_with_structure import make_serializable_object
from chainer_npz_with_structure import save_npz_with_structure

parser = argparse.ArgumentParser()
parser.add_argument('--output_model', default='faster-rcnn-vgg16-voc07.npz')
parser.add_argument('--output_json', default='faster-rcnn-vgg16-voc07.json')
parser.add_argument('--pretrained_model', default='voc07',
                    help = 'the pretrained model. See http://chainercv.readthedocs.io/en/stable/reference/links/faster_rcnn.html '
)
args = parser.parse_args()

print("Prepare the pretrained model.")
model = make_serializable_object(
    FasterRCNNVGG16,
    {
        'n_fg_class': len(voc_bbox_label_names),
        'pretrained_model': args.pretrained_model,
    })

print("Save the pretrained model as '%s', which can be loaded by 'chainer_npz_with_structure.load_npz_with_structure()'." % (args.output_model,))
save_npz_with_structure(args.output_model, model)

print("Save the label file as '%s'." % (args.output_json,))
d = dict()
for n, name in enumerate(voc_bbox_label_names):
    d[n] = name

with open(args.output_json, 'w') as fp:
    json.dump(d, fp, indent=4)
