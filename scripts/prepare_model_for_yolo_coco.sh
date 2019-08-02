#! /bin/bash

# Save the model and related files in the current directory.
DEST_DIR=$PWD

DARKNET_LIBRARY=libdarknet.so
# To use the library compiled by 'CMakeLists.txt' in the package 'darknet_ros',
# specify 'libdarknet_ros_lib.so' as DARKNET_LIBRARY=libdarknet_ros_lib.so .
if [ ! -z "$1" ]; then
    DARKNET_LIBRARY=$1
fi

echo "Use \"${DARKNET_LIBRARY}\" as a library file of Darknet."
echo "Now try to import the library \"${DARKNET_LIBRARY}\"..."
cat <<EOF | python -
import ctypes
import sys
darknet_library = "${DARKNET_LIBRARY}"
try:
    lib = ctypes.CDLL(darknet_library, ctypes.RTLD_GLOBAL)
except OSError as err:
    print('OS error: %s' % (err,))
    sys.exit(1)
EOF
if [ $? -eq 0 ]; then
    echo "Succeeded to import the darknet library \"${DARKNET_LIBRARY}\"."
else
    echo "Failed to import the darknet library \"${DARKNET_LIBRARY}\"."
    echo "A valid library name or a path should be given as an argument of this script."
    exit 1
fi

echo
echo "Download files related to pre-trained models..."

BASENAME=${0##*/}

TMPDIR=`mktemp -t -d ${BASENAME}.XXXXXXXXXX`

pushd ${TMPDIR}

# Generate a JSON file including class names.
mkdir -p cfg
mkdir -p data
wget -P cfg https://github.com/pjreddie/darknet/raw/master/cfg/coco.data
wget -P data https://github.com/pjreddie/darknet/raw/master/data/coco.names
rosrun detect_object convert_darknet_metadata_to_json.py \
       --darknet_library ${DARKNET_LIBRARY} \
       --input_metadata cfg/coco.data --output_json ${DEST_DIR}/coco.json

# Convert a YOLOv2 model trained with the COCO dataset into the format
# for Chainer.
wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov2.cfg
wget https://pjreddie.com/media/files/yolov2.weights
rosrun detect_object convert_darknet_to_npz.py \
       --darknet_library ${DARKNET_LIBRARY} \
       --model_type yolo_v2 --n_fg_class 80 \
       --input_darknet_model yolov2.weights --input_darknet_cfg yolov2.cfg \
       --output ${DEST_DIR}/yolo-v2-coco.npz

# Convert a YOLOv3 model trained with the COCO dataset into the format
# for Chainer.
wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
rosrun detect_object convert_darknet_to_npz.py \
       --darknet_library ${DARKNET_LIBRARY} \
       --model_type yolo_v3 --n_fg_class 80 \
       --input_darknet_model yolov3.weights --input_darknet_cfg yolov3.cfg \
       --output ${DEST_DIR}/yolo-v3-coco.npz

# Return to the original directory.
popd

# Remove the temporary directory.
rm -r ${TMPDIR}
