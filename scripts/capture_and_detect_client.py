#!/usr/bin/env python

import sys
import rospy
import cv2
import cv_bridge

import argparse
import numpy as np

import detect_object.srv

def visualize_result_onto(cv_image, result):
    for n, region in enumerate(result.regions):
        x0 = region.x_offset
        y0 = region.y_offset
        x1 = region.x_offset + region.width - 1
        y1 = region.y_offset + region.height - 1
        cv2.rectangle(cv_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        label_str = '%.2f: %s' % (result.scores[n], result.names[n])
        text_config = {
            'text': label_str,
            'fontFace': cv2.FONT_HERSHEY_PLAIN,
            'fontScale': 1,
            'thickness': 1,
        }
        size, baseline = cv2.getTextSize(**text_config)
        cv2.rectangle(
            cv_image, (x0, y0), (x0 + size[0], y0 + size[1]),
            (255, 255, 255), cv2.FILLED
        )
        cv2.putText(
            cv_image,
            org = (x0, y0 + size[1]),
            color = (255, 0, 0),
            **text_config
        )
    return cv_image

filename = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--service_name', default='capture_and_detect',
                        help = 'name of "capture_and_detect" service')
    parser.add_argument('--topic', default='', help = 'source topic')
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
        visualize_result_onto(cv_image, res)
        cv2.imshow(filename, cv_image)

    if args.display:
        print('Press any key on an image window to finish the program.')
        while True:
            k = cv2.waitKey(1000)
            if k != -1:
                cv2.destroyAllWindows()
                break
