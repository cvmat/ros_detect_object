#!/usr/bin/env python
#
# Based on https://github.com/chainer/chainercv/blob/master/examples/yolo/darknet2npz.py
#
import argparse
import numpy as np
import sys

import chainer
from chainer.links import Convolution2D
from chainer import serializers

import chainercv
from chainercv.links import Conv2DBNActiv

from chainer_npz_with_structure import make_serializable_object
from chainer_npz_with_structure import save_npz_with_structure

def load_param(file, param):
    if isinstance(param, chainer.Variable):
        param = param.array
    param[:] = np.fromfile(file, dtype=np.float32, count=param.size) \
                 .reshape(param.shape)


def load_link(file, link):
    if isinstance(link, Convolution2D):
        load_param(file, link.b)
        load_param(file, link.W)
    elif isinstance(link, Conv2DBNActiv):
        load_param(file, link.bn.beta)
        load_param(file, link.bn.gamma)
        load_param(file, link.bn.avg_mean)
        load_param(file, link.bn.avg_var)
        load_param(file, link.conv.W)
    elif isinstance(link, chainer.ChainList):
        for l in link:
            load_link(file, l)


def reorder_loc(conv, n_fg_class):
    # xy -> yx
    for data in (conv.W.array, conv.b.array):
        data = data.reshape(
            (-1, 4 + 1 + n_fg_class) + data.shape[1:])
        data[:, [1, 0, 3, 2]] = data[:, :4].copy()


def load_yolo_v2(file, model):
    load_link(file, model.extractor)
    load_link(file, model.subnet)

    reorder_loc(model.subnet, model.n_fg_class)


def load_yolo_v3(file, model):
    for i, link in enumerate(model.extractor):
        load_link(file, link)
        if i in {33, 39, 45}:
            subnet = model.subnet[(i - 33) // 6]
            load_link(file, subnet)

    for subnet in model.subnet:
        reorder_loc(subnet[-1], model.n_fg_class)

def read_version_of_format(f):
    major = np.fromfile(f, dtype=np.int32, count=1)
    minor = np.fromfile(f, dtype=np.int32, count=1)
    return major, minor

def update_format_of_darknet_model(output_filename, input_filename, darknet_library, darknet_cfg_file):
    import ctypes
    try:
        lib = ctypes.CDLL(darknet_library, ctypes.RTLD_GLOBAL)
    except OSError as err:
        print('OS error: %s' % (err,))
        print('Failed to import the darknet library "%s"' % (darknet_library,))
        return False

    load_net = lib.load_network
    load_net.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    load_net.restype = ctypes.c_void_p

    save_weights = lib.save_weights
    save_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    net = load_net(darknet_cfg_file, input_filename, 0)
    save_weights(net, output_filename)
    return True

def load_darknet_model(filename, model_type, number_of_foreground_classes, retry_with_update_if_possible=False, darknet_library='libdarknet.so',  darknet_cfg_filename=''):
    if model_type == 'yolo_v2':
        model = make_serializable_object(
            chainercv.links.YOLOv2,
            constructor_args = {
                'n_fg_class': number_of_foreground_classes,
            }
        )
    elif model_type == 'yolo_v3':
        model = make_serializable_object(
            chainercv.links.YOLOv3,
            constructor_args = {
                'n_fg_class': number_of_foreground_classes,
            }
        )

    with chainer.using_config('train', False):
        model(np.empty((1, 3, model.insize, model.insize), dtype=np.float32))

    model_is_old_format = False
    with open(filename, mode='rb') as f:
        major = np.fromfile(f, dtype=np.int32, count=1)
        minor = np.fromfile(f, dtype=np.int32, count=1)
        np.fromfile(f, dtype=np.int32, count=1)  # revision
        if major == 0 and minor <= 1:
            model_is_old_format = True
            print(
                'The file "%s" is written in an old format, where (major, minor) == (%d, %d) '
                % (filename, major, minor)
            )
            sys.stdout.flush()
            model = None
        else:
            print(
                'The file "%s" is written in a new format, where (major, minor) == (%d, %d) '
                % (filename, major, minor)
            )
            assert(major * 10 + minor >= 2 and major < 1000 and minor < 1000)
            model_is_old_format = False
            np.fromfile(f, dtype=np.int64, count=1)  # seen
            if model_type == 'yolo_v2':
                load_yolo_v2(f, model)
            elif model_type == 'yolo_v3':
                load_yolo_v3(f, model)

    if model_is_old_format and retry_with_update_if_possible:
        import tempfile
        import os
        print(
            'Try to convert "%s" to the new format and load it.'
            % (filename,)
        )
        sys.stdout.flush()
        fileno, tmp_path = tempfile.mkstemp()
        try:
            succeeded = update_format_of_darknet_model(
                tmp_path, filename, darknet_library, darknet_cfg_filename
            )
            if succeeded:
                print('Succeeded to update "%s" and save the updated version into "%s".'
                      % (filename, tmp_path))
                sys.stdout.flush()
                model = load_darknet_model(
                    tmp_path, model_type, number_of_foreground_classes,
                    retry_with_update_if_possible=False
                )
            else:
                print('Failed to convert "%s".' % (filename,))
                sys.stdout.flush()
        finally:
            # remove the temporary file
            os.remove(tmp_path)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type', choices=('yolo_v2', 'yolo_v3'),
        default='yolo_v2')
    parser.add_argument('--n_fg_class', type=int, default=80)
    parser.add_argument('--darknet_library', default='libdarknet.so')
    parser.add_argument('--input_darknet_model', default='yolov2.weights')
    parser.add_argument('--input_darknet_cfg', default='yolov2.cfg')
    parser.add_argument('--output')
    args = parser.parse_args()

    input_model_filename = args.input_darknet_model
    input_cfg_filename = args.input_darknet_cfg
    output_filename = args.output
    print('Loading a darknet model from "%s".' % (input_model_filename, ))
    sys.stdout.flush()
    model = load_darknet_model(
        input_model_filename, args.model_type, args.n_fg_class,
        retry_with_update_if_possible=True,
        darknet_library=args.darknet_library,
        darknet_cfg_filename=args.input_darknet_cfg,
    )
    if model:
        print('Saving a darknet model to "%s".' % (output_filename, ))
        save_npz_with_structure(output_filename, model)
    else:
        print('Failed to load a model from "%s".' % (input_model_filename, ))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
