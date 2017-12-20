#!/usr/bin/env python
import roslib
roslib.load_manifest('my_package')
import sys
import rospy
import cv2

import cv_bridge
import sensor_msgs.msg

import argparse
import numpy as np
import chainercv
import matplotlib
import matplotlib.pyplot as plot

import my_package.srv


class image_converter:
    def __init__(self, input_topic, output_topic, detect_object_service):
        self.detect_object_service = detect_object_service
        self.image_pub = rospy.Publisher(
            output_topic, sensor_msgs.msg.Image, queue_size=10)
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(
            input_topic, sensor_msgs.msg.Image,
            self.callback)

    def callback(self,data):
        try:
            from timeit import default_timer as timer
            start = timer()
            res = self.detect_object_service(data)
            end = timer()
            print('prediction finished. (%f [sec])' % (end - start, ))
        except cv_bridge.CvBridgeError as e:
            print(e)

        cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        bboxes = []
        for region in res.regions:
            bboxes.append([
                region.x_offset, region.y_offset,
                region.x_offset + region.width - 1,
                region.y_offset + region.height - 1
            ])
        bboxes = np.array(bboxes)
        labels = np.array(res.labels)
        scores = np.array(res.scores)
        bbox_label_names = chainercv.datasets.voc_bbox_label_names + tuple(
            ['nippn_hamburger', 'nippn_broccoli']
        )
        np_img = np.array([cv_image[:,:,2],
                           cv_image[:,:,1],
                           cv_image[:,:,0]])
        matplotlib.pyplot.clf()
        fig = matplotlib.pyplot.figure()
        chainercv.visualizations.vis_bbox(
            np_img, bboxes, labels, scores, label_names=bbox_label_names,
            ax = matplotlib.pyplot.axes()
        )
        fig.canvas.draw()
        img_data = np.fromstring(
            fig.canvas.tostring_rgb(), dtype=np.uint8, sep=''
        )
        cv_image = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except cv_bridge.CvBridgeError as e:
            print(e)

def main(args):
    print("Wait for the service 'detect_object'...")
    rospy.wait_for_service('detect_object')
    print("Finished.")
    detect_object_service = rospy.ServiceProxy(
        'detect_object', my_package.srv.DetectObject)

    input_topic = rospy.resolve_name("input")
    output_topic = rospy.resolve_name("output")
    ic = image_converter(input_topic, output_topic, detect_object_service)
    rospy.init_node('monitor_and_detect', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


