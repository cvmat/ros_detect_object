#!/usr/bin/env python
import roslib; roslib.load_manifest('detect_object')
import cv2
import cv_bridge
import rospy
import sensor_msgs.msg

import argparse
import chainer
import numpy as np
import sys
from timeit import default_timer as timer
from chainer_npz_with_structure import load_npz_with_structure

import detect_object.srv

gpu_device_id = None

def handle_detect(req):
    global bridge, model, gpu_device_id
    cv_image = bridge.imgmsg_to_cv2(req.image, "rgb8")
    (rows,cols,channels) = cv_image.shape
    img = np.array([cv_image[:,:,0],cv_image[:,:,1],cv_image[:,:,2]])
    img = img.astype(np.float32)
    start = timer()
    if not (gpu_device_id is None):
        import cupy
        with cupy.cuda.Device(gpu_device_id):
            bboxes, labels, scores = model.predict([img])
    else:
        bboxes, labels, scores = model.predict([img])
    end = timer()
    print('prediction finished. (%f [sec])' % (end - start, ))
    sys.stdout.flush()
    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(3)
    try:
        res = detect_object.srv.DetectObjectResponse()
        img_height, img_width = cv_image.shape[0:2]
        for bbox, label, score in zip(bboxes[0], labels[0], scores[0]):
            # Ensure that x_offset and y_offset are not negative.
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            roi_param = {
                'y_offset': bbox[0], 'x_offset': bbox[1],
                'height': bbox[2] - bbox[0] + 1,
                'width': bbox[3] - bbox[1] + 1,
                'do_rectify': True
            }
            res.regions.append(sensor_msgs.msg.RegionOfInterest(**roi_param))
            res.scores.append(score)
            res.labels.append(label)
            res.names.append(label_names[label])
    except cv_bridge.CvBridgeError as e:
        print(e)
    return res

def handle_capture_and_detect(req):
    global bridge, model, gpu_device_id
    rospy.loginfo('Received a request {topic: "%s"}.' % (req.topic,))
    # Prepare a response object.
    res = detect_object.srv.CaptureAndDetectResponse()

    # Capture an image from the given topic.
    topics = rospy.get_published_topics()
    source_topic = req.topic
    source_topic_type = None
    for topic_name, topic_type in topics:
        if topic_name == source_topic:
            source_topic_type = topic_type
            break

    if source_topic_type == None:
        res.success = False
        res.error_msg = 'Failed to find the topic [%s].' % (source_topic,)
        rospy.logerr(res.error_msg)
        return res
    elif source_topic_type != 'sensor_msgs/Image':
        res.success = False
        res.error_msg = (
            'The topic [%s] is found but its type is [%s], not [sensor_msgs/Image].'
            % (source_topic, source_topic_type)
        )
        rospy.logerr(res.error_msg)
        return res

    timeout = req.capture_timeout.to_sec()
    if timeout == 0:
        timeout = None
    rospy.loginfo('Waiting for the topic "%s"...' % (source_topic,))
    try:
        input_msg = rospy.wait_for_message(
            source_topic, sensor_msgs.msg.Image,
            timeout = timeout
        )
    except rospy.ROSException as e:
        rospy.logerr(str(e))
        res.success = False
        res.error_msg = str(e)
        return res

    rospy.loginfo(
        'Received an image [%dx%d].' % (input_msg.width, input_msg.height)
    )

    try:
        cv_image = bridge.imgmsg_to_cv2(input_msg, "rgb8")
    except cv_bridge.CvBridgeError as e:
        rospy.logerr(e)
        res.success = False
        res.error_msg = e
        return res

    (rows,cols,channels) = cv_image.shape
    img = np.array([cv_image[:,:,0],cv_image[:,:,1],cv_image[:,:,2]])
    img = img.astype(np.float32)
    start = timer()
    if not (gpu_device_id is None):
        import cupy
        with cupy.cuda.Device(gpu_device_id):
            bboxes, labels, scores = model.predict([img])
    else:
        bboxes, labels, scores = model.predict([img])
    end = timer()
    rospy.loginfo('prediction finished. (%f [sec])' % (end - start, ))

    img_height, img_width = cv_image.shape[0:2]
    for bbox, label, score in zip(bboxes[0], labels[0], scores[0]):
        # Ensure that x_offset and y_offset are not negative.
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        roi_param = {
            'y_offset': bbox[0], 'x_offset': bbox[1],
            'height': bbox[2] - bbox[0] + 1,
            'width': bbox[3] - bbox[1] + 1,
            'do_rectify': True
        }
        res.regions.append(sensor_msgs.msg.RegionOfInterest(**roi_param))
        res.scores.append(score)
        res.labels.append(label)
        res.names.append(label_names[label])
    res.success = True
    if req.return_image_data:
        res.image = input_msg
    else:
        for attr in [ 'header', 'height', 'width']:
            setattr(res.image, attr, getattr(input_msg, attr))
    return res


def start_server(service_name_of_detect_object, service_name_of_capture_and_detect):
    print('Invoke rospy.Service().')
    sys.stdout.flush()
    rospy.Service(service_name_of_detect_object,
                  detect_object.srv.DetectObject, handle_detect)
    rospy.Service(
        service_name_of_capture_and_detect,
        detect_object.srv.CaptureAndDetect, handle_capture_and_detect
    )
    print "Ready to detect objects."
    sys.stdout.flush()
    rospy.spin()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ROS services for object detection.',
    )
    parser.add_argument('--detection_service_name', default='',
                        help = 'name of "detect_object" service')
    parser.add_argument('--service_name_of_detect_object',
                        default='detect_object',
                        help = 'name of "detect_object" service')
    parser.add_argument('--service_name_of_capture_and_detect',
                        default='capture_and_detect',
                        help = 'name of "capture_and_detect" service')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--label_file', default='',
                        help = 'JSON file of label names')
    parser.add_argument('--model', default='',
                        help = "NPZ file of a trained Faster R-CNN model generated by 'chainer_npz_with_structure.load_npz_with_structure()'.")
    parser.add_argument('--node_name', default='detect_object_server',
                        help = 'node name')
    parser.add_argument('--tcpros_port', type=int, default=0,
                        help = 'port for services (default: auto)')
    parser.add_argument('--xmlrpc_port', type=int, default=0,
                        help = 'port for XML-RPC (default: auto)')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    print('Invoke rospy.init_node().')
    sys.stdout.flush()
    rospy.init_node(
        args.node_name,
        xmlrpc_port=args.xmlrpc_port, tcpros_port=args.tcpros_port
    )

    rospy.loginfo('Load %s...' % (args.model,))
    model = load_npz_with_structure(args.model)
    rospy.loginfo('Finished.')

    if 'n_fg_class' in dir(model):
        label_names = ['label%d' % n for n in range(model.n_fg_class)]
    else:
        label_names = ['label%d' % n for n in range(model.n_class - 1)]
    if args.label_file != '':
        import json
        with open(args.label_file, 'r') as fp:
            json_data = json.load(fp)
        label_names = dict()
        for k, v in json_data.items():
            label_names[int(k)] = v

    if args.gpu >= 0:
        rospy.loginfo('Invoke model.to_gpu().')
        gpu_device_id = args.gpu
        import cupy
        with cupy.cuda.Device(gpu_device_id):
            model.to_gpu(gpu_device_id)
        rospy.loginfo('Finished.')

    if args.detection_service_name == '':
        args.detection_service_name = args.service_name_of_detect_object

    #
    rospy.loginfo('Node name: %s' % (args.node_name, ))
    rospy.loginfo('Service name for "detect_object": %s'
                  % (args.detection_service_name, ))
    rospy.loginfo('Service name for "capture_and_detect": %s'
                  % (args.service_name_of_capture_and_detect, ))
    rospy.loginfo('Listen to %s/tcp for XML-RPC and %s/tcp for services.'
                  % (args.xmlrpc_port, args.tcpros_port, ))
    rospy.loginfo('(0 means that a port will be automatically determined.)')
    bridge = cv_bridge.CvBridge()
    start_server(
        args.detection_service_name, args.service_name_of_capture_and_detect
    )

