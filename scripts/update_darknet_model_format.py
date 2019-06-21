#!/usr/bin/env python
import argparse
import ctypes
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Update format of a darknet model by loading/saving it.',
        epilog='''
EXAMPLE:
python update_darknet_model_format.py --input_model yolov2.weights --input_cfg yolov2.cfg --output_model yolov2_new.weights
'''
    )
    parser.add_argument('--darknet_library', default='libdarknet.so')
    parser.add_argument('--input_model', default='yolov2.weights')
    parser.add_argument('--input_cfg', default='yolov2.cfg')
    parser.add_argument('--output_model', default='yolov2_new.weights')

    parsed_args = sys.argv
    try:
        import rospy
        parsed_args = rospy.myargv(argv=sys.argv)
    except ImportError:
        pass

    args = parser.parse_args(parsed_args[1:])

    lib = ctypes.CDLL(args.darknet_library, ctypes.RTLD_GLOBAL)
    load_net = lib.load_network
    load_net.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    load_net.restype = ctypes.c_void_p

    save_weights = lib.save_weights
    save_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    net = load_net(args.input_cfg, args.input_model, 0)
    save_weights(net, args.output_model)
