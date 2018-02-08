#!/usr/bin/env python

import sys
import rospy
import cv2
import cv_bridge

import argparse
import numpy as np

import detect_object.srv

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
    parser.add_argument('--display', action='store_true',
                        help='display images')
    parser.add_argument('filenames', metavar='FILE', nargs='+',
                        help='a filename of a image')
    args = parser.parse_args()
    if args.display:
        import chainercv
        import matplotlib

    print('Waiting for the service "%s"...' % (args.detection_service_name,))
    rospy.wait_for_service(args.detection_service_name)
    detect_object_service = rospy.ServiceProxy(
        args.detection_service_name, detect_object.srv.DetectObject)
    print('Connected to the service "%s".' % (args.detection_service_name,))
    for filename in args.filenames:
        print "Requesting %s"%(filename)
        img = cv2.imread(filename)
        from timeit import default_timer as timer
        start = timer()
        res = detect_object_client(detect_object_service, img)
        end = timer()
        print('prediction finished. (%f [sec])' % (end - start, ))
        if not args.display:
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
        else:
            bboxes = []
            for region in res.regions:
                bboxes.append([
                    region.y_offset, region.x_offset,
                    region.y_offset + region.height - 1,
                    region.x_offset + region.width - 1
                ])
            bboxes = np.array(bboxes)
            labels = np.array(res.labels)
            scores = np.array(res.scores)
            names = np.array(res.names)
            #
            # Prepare `label_names` for `chainercv.visualizations.vis_bbox()`
            # that assumes that `label_names` includes all label names.
            #
            max_label = 0
            if 0 < len(labels):
                max_label = labels.max()
            label_names = [''] * (1 + max_label)
            for label_id, label_name in zip(labels, names):
                label_names[label_id] = label_name
            #
            #
            np_img = np.array([img[:,:,2],img[:,:,1],img[:,:,0]])
            chainercv.visualizations.vis_bbox(
                np_img, bboxes, labels, scores, label_names=label_names)
    if args.display:
        matplotlib.pyplot.show()

