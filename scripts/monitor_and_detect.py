#!/usr/bin/env python
import roslib
roslib.load_manifest('detect_object')
import sys
import rospy
import cv2

import cv_bridge
import sensor_msgs.msg

import argparse
import numpy as np
import chainercv

import detect_object.srv

class image_converter:
    def __init__(self, input_topic, output_topic, detect_object_service):
        self.detect_object_service = detect_object_service
        self.image_pub = rospy.Publisher(
            output_topic, sensor_msgs.msg.Image, queue_size=10)
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(
            input_topic, sensor_msgs.msg.Image,
            self.callback)

    def callback(self, data):
        try:
            from timeit import default_timer as timer
            start = timer()
            res = self.detect_object_service(data)
            end = timer()
            print('Finished detection. (%f [sec])' % (end - start, ))
            sys.stdout.flush()
        except cv_bridge.CvBridgeError as e:
            print(e)
            return

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        for n, region in enumerate(res.regions):
            x0 = region.x_offset
            y0 = region.y_offset
            x1 = region.x_offset + region.width - 1
            y1 = region.y_offset + region.height - 1
            cv2.rectangle(cv_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
            label_str = '%.2f: %s' % (res.scores[n], res.names[n])
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
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)
        try:
            self.image_pub.publish(
                self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            )
        except cv_bridge.CvBridgeError as e:
            print(e)

def main(args):
    print("Wait for the service 'detect_object'...")
    sys.stdout.flush()
    rospy.wait_for_service('detect_object')
    print("Finished.")
    sys.stdout.flush()
    detect_object_service = rospy.ServiceProxy(
        'detect_object', detect_object.srv.DetectObject)
    input_topic = rospy.resolve_name("input")
    output_topic = rospy.resolve_name("output")
    ic = image_converter(input_topic, output_topic, detect_object_service)
    print("Invoke rospy.init_node().")
    sys.stdout.flush()
    rospy.init_node('monitor_and_detect', anonymous=True)
    try:
        print("Invoke rospy.spin().")
        sys.stdout.flush()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


