#!/usr/bin/env python

import sys
import rospy
import cv2
import cv_bridge

import argparse
import numpy as np

import detect_object.srv
import sensor_msgs

bridge = cv_bridge.CvBridge()

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_service_name', default='detect_object',
                        help = 'name of detection service')
    parser.add_argument('--output', default='output.png',
                        help = 'an output filename')
    parser.add_argument('--input', default='',
                        help='an input topic')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    input_topic = args.input
    output_filename = args.output

    rospy.init_node('detect_snapshot', anonymous=True)
    topics = rospy.get_published_topics()
    if input_topic == '':
        rospy.logwarn('No input topics is given.')
        rospy.logwarn('Searching a topic with the type [sensor_msgs/Image]...')
        for topic_str, type_str in topics:
            if input_topic == '' and type_str == 'sensor_msgs/Image':
                input_topic = topic_str
                rospy.loginfo('Found "%s". Use it.' % (topic_str,))

    if input_topic == '':
        rospy.logerr('Failed to found an input topic.')
        return

    print('Waiting for the topic "%s"...' % (input_topic,))
    sys.stdout.flush()
    input_msg = rospy.wait_for_message(input_topic, sensor_msgs.msg.Image)
    print('Received an image [%dx%d].' % (input_msg.width, input_msg.height))
    sys.stdout.flush()

    print('Waiting for the service "%s"...' % (args.detection_service_name,))
    rospy.wait_for_service(args.detection_service_name)
    detect_object_service = rospy.ServiceProxy(
        args.detection_service_name, detect_object.srv.DetectObject)
    print('Connected to the service "%s".' % (args.detection_service_name,))

    try:
        from timeit import default_timer as timer
        start = timer()
        res = detect_object_service(input_msg)
        end = timer()
        print('Finished prediction. (%f [sec])' % (end - start, ))
    except rospy.ServiceException, e:
        rospy.logerr("Service call failed: %s" % e)

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

    img = bridge.imgmsg_to_cv2(input_msg, "bgr8")
    visualize_result_onto(img, res)
    print('Saving %s' % (output_filename,))
    cv2.imwrite(output_filename, img)
    return

if __name__ == "__main__":
    main()
