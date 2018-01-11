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



def handle_detect(req):
    global bridge, model
    cv_image = bridge.imgmsg_to_cv2(req.image, "rgb8")
    (rows,cols,channels) = cv_image.shape
    img = np.array([cv_image[:,:,0],cv_image[:,:,1],cv_image[:,:,2]])
    from timeit import default_timer as timer
    start = timer()
    bboxes, labels, scores = model.predict([img])
    end = timer()
    print('prediction finished. (%f [sec])' % (end - start, ))
    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(3)
    try:
        res = detect_object.srv.DetectObjectResponse()
        img_height, img_width = cv_image.shape[0:2]
        for bbox, label, score in zip(bboxes[0], labels[0], scores[0]):
            roi_param = {
                'x_offset': bbox[0], 'y_offset': bbox[1],
                'width': bbox[2] - bbox[0] + 1,
                'height': bbox[3] - bbox[1] + 1,
                'do_rectify': True
            }
            res.regions.append(sensor_msgs.msg.RegionOfInterest(**roi_param))
            res.scores.append(score)
            res.labels.append(label)
            res.names.append(label_names[label])
    except cv_bridge.CvBridgeError as e:
        print(e)
    return res

def detect_object_server():
    rospy.init_node('detect_object_server',
                    xmlrpc_port=60000, tcpros_port=60001)
    s = rospy.Service('detect_object',
                      detect_object.srv.DetectObject, handle_detect)
    print "Ready to detect objects."
    rospy.spin()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', default='',
                        help = 'a NPZ file of a trained Faster R-CNN model')
    parser.add_argument('--label_file', default='',
                        help = 'a JSON file of label names')
    args = parser.parse_args()

    print('Load %s...' % (args.model,))
    sys.stdout.flush()
    model = load_npz_with_structure(args.model)
    print('Finished.')
    sys.stdout.flush()

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
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        print('Finished.')
        sys.stdout.flush()

    bridge = cv_bridge.CvBridge()
    detect_object_server()

