#!/usr/bin/env bash
python gen_result_file.py --name $1 --list /home/hezheqi/data/coco_text/$3/img_list.txt --type coco_text --iter $2 --imdb $3 --pkl /home/hezheqi/Project/dev/frame_regression --poly --net res152
