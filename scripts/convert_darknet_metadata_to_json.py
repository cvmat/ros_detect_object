#!/usr/bin/env python
import argparse
import ctypes
import json
import sys

class METADATA(ctypes.Structure):
    _fields_ = [("classes", ctypes.c_int),
                ("names", ctypes.POINTER(ctypes.c_char_p))]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a metadata for a darknet model into a JSON label file which can be used by 'detect.py' or 'detect_once.py'.",
        epilog='''
EXAMPLE:
python convert_darknet_metadata_to_json.py --input_metadata cfg/coco.data --output_json coco.json
'''
    )
    parser.add_argument('--darknet_library', default='libdarknet.so')
    parser.add_argument('--input_metadata', default='cfg/coco.data')
    parser.add_argument('--output_json', default='coco.json')

    parsed_args = sys.argv
    try:
        import rospy
        parsed_args = rospy.myargv(argv=sys.argv)
    except ImportError:
        pass

    args = parser.parse_args(parsed_args[1:])

    lib = ctypes.CDLL(args.darknet_library, ctypes.RTLD_GLOBAL)
    lib.get_metadata.argtypes = [ctypes.c_char_p]
    lib.get_metadata.restype = METADATA
    meta = lib.get_metadata(args.input_metadata)

    print("Saving the label file as '%s'." % (args.output_json,))
    d = dict()
    for i in range(meta.classes):
        d[i] = meta.names[i]

    with open(args.output_json, 'w') as fp:
        json.dump(d, fp, indent=4)

    print("Finished.")
    sys.stdout.flush()
