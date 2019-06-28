#!/usr/bin/env python
import argparse
import chainer
import chainercv
import numpy as np
import sys
from timeit import default_timer as timer
from chainer_npz_with_structure import load_npz_with_structure

gpu_device_id = None

def detect_object(model, cv_image):
    raw_bboxes, raw_labels, raw_scores = model.predict([cv_image])
    #
    result = []
    for bbox, label, score in zip(raw_bboxes[0], raw_labels[0], raw_scores[0]):
        # Ensure that x_offset and y_offset are not negative.
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        region_info = {
            'y_offset': bbox[0], 'x_offset': bbox[1],
            'height': bbox[2] - bbox[0] + 1,
            'width': bbox[3] - bbox[1] + 1,
            'score': score,
            'label': label,
            'name': label_names[label],
        }
        result.append(region_info)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detect objects in given images.',
        epilog='''
EXAMPLE:
python detect_once.py --label_file faster-rcnn-vgg16-voc07.json --model faster-rcnn-vgg16-voc07.npz --gpu 0 input1.png input2.png input3.png
'''
    )
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--label_file', default='',
                        help = 'JSON file of label names')
    parser.add_argument('--model', default='',
                        help = 'NPZ file of a trained Faster R-CNN model')
    parser.add_argument('filenames', metavar='FILE', nargs='+',
                        help='a filename of a image')
    #args = parser.parse_args(sys.argv)
    args = parser.parse_args()

    print('Load %s...' % (args.model,))
    sys.stdout.flush()
    from timeit import default_timer as timer
    start = timer()
    model = load_npz_with_structure(args.model)
    end = timer()
    print('Finished. (%f [sec])' % (end - start, ))
    sys.stdout.flush()

    if 'n_fg_class' in dir(model):
        label_names = ['label%d' % n for n in range(model.n_fg_class)]
    else:
        label_names = ['label%d' % n for n in range(model.n_class - 1)]
    if args.label_file != '':
        import json
        with open(args.label_file, 'r') as fp:
            json_data = json.load(fp)
        label_names = dict()
        for k, v in json_data.items():
            label_names[int(k)] = v

    if args.gpu >= 0:
        print('Invoke model.to_gpu().')
        sys.stdout.flush()
        gpu_device_id = args.gpu
        import cupy
        with cupy.cuda.Device(gpu_device_id):
            model.to_gpu(gpu_device_id)
        print('Finished.')
        sys.stdout.flush()

    #
    #
    for filename in args.filenames:
        print('Requesting %s' % (filename, ))
        sys.stdout.flush()
        #
        img = chainercv.utils.read_image(filename)
        #
        from timeit import default_timer as timer
        start = timer()
        if args.gpu >= 0:
            with cupy.cuda.Device(gpu_device_id):
                result = detect_object(model, img)
        else:
            result = detect_object(model, img)
        end = timer()
        print('prediction finished. (%f [sec])' % (end - start, ))
        sys.stdout.flush()
        #
        print('regions:')
        for region in result:
            print(
                '{'+
                ', '.join(['%s: %s' % (k, v) for k, v in region.items()])
                +'}'
            )
    #
