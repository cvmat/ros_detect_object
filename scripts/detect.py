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
from chainer_npz_with_structure import load_npz_with_structure

import detect_object.srv

gpu_device_id = None

def handle_detect(req):
    global bridge, model, gpu_device_id
    cv_image = bridge.imgmsg_to_cv2(req.image, "rgb8")
    (rows,cols,channels) = cv_image.shape
    img = np.array([cv_image[:,:,0],cv_image[:,:,1],cv_image[:,:,2]])
    img = img.astype(np.float32)
    from timeit import default_timer as timer
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

def detect_object_server(node_name, detection_service_name, xmlrpc_port, tcpros_port):
    print('Invoke rospy.init_node().')
    sys.stdout.flush()
    rospy.init_node(node_name,
                    xmlrpc_port=xmlrpc_port, tcpros_port=tcpros_port)
    print('Invoke rospy.Service().')
    sys.stdout.flush()
    s = rospy.Service(detection_service_name,
                      detect_object.srv.DetectObject, handle_detect)
    print "Ready to detect objects."
    sys.stdout.flush()
    rospy.spin()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_service_name', default='detect_object',
                        help = 'name of detection service')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--label_file', default='',
                        help = 'JSON file of label names')
    parser.add_argument('--model', default='',
                        help = 'NPZ file of a trained Faster R-CNN model')
    parser.add_argument('--node_name', default='detect_object_server',
                        help = 'node name')
    parser.add_argument('--tcpros_port', type=int, default=0,
                        help = 'port for services (default: auto)')
    parser.add_argument('--xmlrpc_port', type=int, default=0,
                        help = 'port for XML-RPC (default: auto)')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    print('Load %s...' % (args.model,))
    sys.stdout.flush()
    model = load_npz_with_structure(args.model)
    print('Finished.')
    sys.stdout.flush()

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
        print('Invoke model.to_gpu().')
        sys.stdout.flush()
        gpu_device_id = args.gpu
        import cupy
        with cupy.cuda.Device(gpu_device_id):
            model.to_gpu(gpu_device_id)
        print('Finished.')
        sys.stdout.flush()

    print('Node name: %s' % (args.node_name, ))
    print('Detection service name: %s' % (args.detection_service_name, ))
    print('Listen to %s/tcp for XML-RPC and %s/tcp for services.'
          % (args.xmlrpc_port, args.tcpros_port, ))
    print('(0 means that a port will be automatically determined.)')
    bridge = cv_bridge.CvBridge()
    detect_object_server(
        args.node_name, args.detection_service_name,
        args.xmlrpc_port, args.tcpros_port
    )

