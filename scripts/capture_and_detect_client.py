#!/usr/bin/env python

import sys
import rospy
import cv2
import cv_bridge

import argparse
import numpy as np

import detect_object.srv
import util

filename = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--service_name', default='capture_and_detect',
                        help = 'name of "capture_and_detect" service')
    parser.add_argument('--topic', default='', help = 'source topic')
    parser.add_argument('--label_format', default='%(score).2f: %(name)s',
                        help = 'format of a label text in visualized result')
    parser.add_argument('--display', action='store_true',
                        help='display images')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    print('Waiting for the service "%s"...' % (args.service_name,))
    rospy.wait_for_service(args.service_name)
    service = rospy.ServiceProxy(
        args.service_name, detect_object.srv.CaptureAndDetect
    )
    print('Connected to the service "%s".' % (args.service_name,))

    print(
        'Requesting to capture an image from [%s] and detect objects in it.'
        % (args.topic,)
    )

    from timeit import default_timer as timer
    start = timer()

    try:
        msg = detect_object.srv.CaptureAndDetectRequest()
        msg.topic = args.topic
        msg.capture_timeout = rospy.Duration(1.0)
        msg.return_image_data = args.display
        res = service(msg)
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

    end = timer()
    print('Finished. (%f [sec])' % (end - start, ))
    print('len(res.image.data): %d' % (len(res.image.data),))
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
        bridge = cv_bridge.CvBridge()
        cv_image = bridge.imgmsg_to_cv2(res.image, "bgr8")
        util.visualize_result_onto(cv_image, res, args.label_format)
        cv2.imshow(filename, cv_image)

    if args.display:
        print('Press any key on an image window to finish the program.')
        while True:
            k = cv2.waitKey(1000)
            if k != -1:
                cv2.destroyAllWindows()
                break
