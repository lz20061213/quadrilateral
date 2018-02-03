#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  frame)
    TRAIN_IMDB="frame_train"
    TEST_IMDB="frame_test"
    PT_DIR="frame"
    ITERS=100000
    ;;
	db42)
    TRAIN_IMDB="frame_train"
    TEST_IMDB="frame_test_dragonball"
    PT_DIR="frame"
    ITERS=100000
    ;;
	frame_2000)
    TRAIN_IMDB="frame_train"
    TEST_IMDB="frame_test_2000"
    PT_DIR="frame"
    ITERS=100000
    ;;
	icdar)
    TRAIN_IMDB="icdar_train"
    TEST_IMDB="icdar_test"
    PT_DIR="icdar"
    ITERS=100000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.25,0.5,1,2,4]"
    ;;
	icdar_1315)
    TRAIN_IMDB="icdar_train+icdar_train_13"
    TEST_IMDB="icdar_test"
    PT_DIR="icdar_1315"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
	icdar_msra)
    TRAIN_IMDB="icdar_train+icdar_train_MSRA-TD500"
    TEST_IMDB="icdar_test"
    PT_DIR="icdar_msra"
    STEPSIZE=60000
    ITERS=100000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.25,0.5,1,2,4]"
    ;;
  icdar_17)
    TRAIN_IMDB="icdar_train_17"
    TEST_IMDB="icdar_test_17"
    PT_DIR="icdar_17"
    ITERS=100000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.25,0.5,1,2,4]"
    ;;
	icdar_171513)
    TRAIN_IMDB="icdar_train_17+icdar_train+icdar_train_13"
    TEST_IMDB="icdar_valid_17"
    PT_DIR="icdar_171513"
    STEPSIZE=60000
    ITERS=100000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.25,0.5,1,2,4]"
    ;;
	icdar_coco)
    TRAIN_IMDB="icdar_train_coco+icdar_train_17+icdar_train"
    TEST_IMDB="icdar_valid_17"
    PT_DIR="icdar_coco"
    STEPSIZE=60000
    ITERS=100000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.25,0.5,1,2,4]"
    ;;
	coco_text)
    TRAIN_IMDB="coco_text_train"
    TEST_IMDB="coco_text_test"
    PT_DIR="ct"
    ITERS=315000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.25,0.5,1,2,4]"
    ;;
	synth)
    TRAIN_IMDB="icdar_train_synth"
    TEST_IMDB="icdar_test_synth"  
    PT_DIR="synth"
    STEPSIZE=350000
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.25,0.5,1,2,4]"
    ;;
	detext)
    TRAIN_IMDB="icdar_train_detext+icdar_valid_detext"
    TEST_IMDB="icdar_test_detext"
    PT_DIR="detext"
    ITERS=2000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.25,0.5,1,2,4]"
    ;;
  coco_text_trainval)
    TRAIN_IMDB="coco_text_train+coco_text_val"
    TEST_IMDB="coco_text_test"
    PT_DIR="ct"
    ITERS=320000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.25,0.5,1,2,4]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ${EXTRA_ARGS}
fi

