#!/usr/bin/env python

import sys
import rospy
import cv2
import cv_bridge

import argparse
import numpy as np

import detect_object.srv
import util

bridge = cv_bridge.CvBridge()
def detect_object_client(detect_object_service, cv_image):
    try:
        msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        res = detect_object_service(msg)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

filename = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_service_name', default='detect_object',
                        help = 'name of detection service')
    parser.add_argument('--output', default='output.png',
                        help = 'an output filename')
    parser.add_argument('--input', metavar='FILE',
                        help='an input filename')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    print('Waiting for the service "%s"...' % (args.detection_service_name,))
    rospy.wait_for_service(args.detection_service_name)
    detect_object_service = rospy.ServiceProxy(
        args.detection_service_name, detect_object.srv.DetectObject)
    print('Connected to the service "%s".' % (args.detection_service_name,))
    input_filename = args.input
    output_filename = args.output
    print('Requesting %s' % (input_filename,))
    img = cv2.imread(input_filename)
    from timeit import default_timer as timer
    start = timer()
    res = detect_object_client(detect_object_service, img)
    end = timer()
    print('prediction finished. (%f [sec])' % (end - start, ))
    print('regions:')
    for region in res.regions:
        print("  "
              "{'x_offset': %d, 'y_offset': %d, "
              "'width': %d, 'height': %d}" % (
                  region.x_offset, region.y_offset,
                  region.width, region.height))
        print('labels: %s' % (res.labels,))
        print('names: %s' % (res.names,))
        print('scores: %s' % (res.scores,))

    util.visualize_result_onto(img, res)
    print('Saving %s' % (output_filename,))
    cv2.imwrite(output_filename, img)

