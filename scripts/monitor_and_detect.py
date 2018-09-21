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
import util

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
        util.visualize_result_onto(cv_image, res)
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
    print("Invoke rospy.init_node().")
    sys.stdout.flush()
    rospy.init_node('monitor_and_detect', anonymous=True)
    input_topic = rospy.resolve_name("input")
    output_topic = rospy.resolve_name("output")
    print("input_topic: %s" % (input_topic,))
    print("output_topic: %s" % (output_topic,))
    sys.stdout.flush()
    ic = image_converter(input_topic, output_topic, detect_object_service)
    try:
        print("Invoke rospy.spin().")
        sys.stdout.flush()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


