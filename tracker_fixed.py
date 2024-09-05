import cv2
import json
import os
import argparse
import torch
import mmcv

import numpy as np
import motmetrics as mm

from time import time
from datetime import timedelta
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from time import time
from datetime import timedelta


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def check_path(path):
    if Path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


parser = argparse.ArgumentParser(description='Tracker options')

parser.add_argument('--gt_path',
                    type=check_path,
                    default=Path('./Datasets/gt/gt.txt'),
                    help='Input dir for images')

parser.add_argument('--imgs_path',
                    type=check_path,
                    default=Path('./Datasets/v2/images/'),
                    help='Input dir for images')

parser.add_argument('--videos_path',
                    type=check_path,
                    default=Path('./Videos/'),
                    help='Input dir for videos')

parser.add_argument('--input_video_path',
                    type=str,
                    default=None,
                    help='Input path for new video')

parser.add_argument('--output_video_path',
                    type=str,
                    default=Path('./output.mp4'),
                    help='Out path for a processed video')

parser.add_argument('-v',
                    type=int,
                    default=None,
                    help='Process a regular video by a given index: 0-5')

parser.add_argument('-vb',
                    type=int,
                    default=None,
                    help='Process a blured video by a given index: 0-18')

parser.add_argument('--model_weights',
                    type=check_path,
                    default=Path('./Models/ultralytics/yolov10x_v2_4_best.pt'),
                    help='YOLO model weights')

parser.add_argument('--tracker',
                    type=str,
                    default='smile',
                    help='Tracker model')

parser.add_argument('--show_video',
                    action=argparse.BooleanOptionalAction,
                    help='Wheither to show video of tracking or not')

parser.add_argument('--metrics',
                    action=argparse.BooleanOptionalAction,
                    help='Count metric or not')

parser.add_argument('--first_n_frames',
                    type=int,
                    default=9000,
                    help='Apply only to first n frames')


args = parser.parse_args()

videos = ['31-03-2024-09%3A34%3A24.mp4',
          '31-03-2024-10%3A13%3A25.mp4',
          '31-03-2024-11%3A05%3A35.mp4',
          '31-03-2024-11%3A48%3A17.mp4',
          '31-03-2024-12%3A32%3A59.mp4',
          '31-03-2024-13%3A19%3A13.mp4' ]

blured_videos = [
    'Смазанные/30-03-2024-08%3A52%3A47.mp4',
    'Смазанные/30-03-2024-09%3A02%3A53.mp4',
    'Смазанные/30-03-2024-09%3A29%3A10.mp4',
    'Смазанные/30-03-2024-09%3A41%3A17.mp4',
    'Смазанные/30-03-2024-10%3A00%3A34.mp4',
    'Смазанные/31-03-2024-05%3A21%3A32.mp4',
    'Смазанные/31-03-2024-07%3A08%3A04.mp4',
    'Смазанные/30-03-2024-08%3A56%3A28.mp4',
    'Смазанные/30-03-2024-09%3A08%3A28.mp4',
    'Смазанные/30-03-2024-09%3A32%3A39.mp4',
    'Смазанные/30-03-2024-09%3A46%3A58.mp4',
    'Смазанные/30-03-2024-11%3A44%3A15.mp4',
    'Смазанные/31-03-2024-05%3A56%3A32.mp4',
    'Смазанные/30-03-2024-08%3A59%3A44.mp4',
    'Смазанные/30-03-2024-09%3A12%3A34.mp4',
    'Смазанные/30-03-2024-09%3A38%3A32.mp4',
    'Смазанные/30-03-2024-09%3A50%3A01.mp4',
    'Смазанные/30-03-2024-13%3A40%3A28.mp4',
    'Смазанные/31-03-2024-06%3A35%3A37.mp4'
    ]



if args.tracker == 'smile':
    from ultralytics.trackers.smiletrack import SMILEtrack
    # from smile_note import SMILEtrack
    from ultralytics.utils import IterableSimpleNamespace

    tracker_args = IterableSimpleNamespace(**
        {
        'tracker_type': 'smiletrack',
        'track_high_thresh': 0.5,
        'track_low_thresh': 0.1,
        'new_track_thresh': 0.6,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'fuse_score': True,
        'gmc_method': 'sparseOptFlow',
        'proximity_thresh': 0.0,
        'appearance_thresh': 0.25,
        'with_reid': True,
        # 'device' : 'cpu',
        # 'reid_default': True,
        # 'weight_path': './ver12.pt',
        # 'mot20': True
        }
        )
    tracker = SMILEtrack(tracker_args)



if args.metrics:
    mm_gt = np.loadtxt(args.gt_path, delimiter=',')

if args.input_video_path is not None:
    video = Path(args.input_video_path)

elif args.vb is not None:
    video = args.videos_path / blured_videos[args.vb]

elif args.v is not None:
    video = args.videos_path / videos[args.v]

else:
    video = args.videos_path / videos[0]


cap = mmcv.VideoReader(str(video))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output_video_path, fourcc,
                      cap.fps, (cap.width, cap.height))

detect_model = YOLO(args.model_weights)
track_history = defaultdict(lambda: defaultdict(list))
mm_reslts = []
break_i = args.first_n_frames

start = time()
for i, frame in enumerate(cap):
    try:
        if i == break_i:
            out.release()
            if args.show_video:
                cv2.destroyAllWindows()
            break
        results = detect_model(frame)
        frame = results[0].plot()
        # if len(results[0].boxes.xyxy) == 0:
        #     continue
        # if results[0].boxes.id is None:
        #     continue
        online_targets = tracker.update(results[0].boxes.cpu(), frame)

        for row in online_targets:
            x1, y1, x2, y2, track_id, conf, cls, ind = row
            if args.metrics:
                mm_reslts.append([i+1, track_id, x1, y1, x2-x1, y2-y1, -1, int(cls), conf])
            history = track_history[int(track_id)]
            history['coord'].append([float(i) for i in [x1, y1, x2, y2]])
            history['class_id'].append(int(cls))
            history['center'].append([float(x1+x2)/2, float(y1+y2)/2])
            if len(history['coord']) > 30:
                history['coord'].pop(0)
                history['class_id'].pop(0)
                history['center'].pop(0)
        if args.show_video:
            cv2.namedWindow('video', 0)
            cv2.imshow(frame, 'video', 1)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except IndexError:
        out.write(frame)
        continue
if out:
    out.release()
cv2.destroyAllWindows()
stop = time()


json_results = json.dumps(track_history)
with open("./tracking_results.json", "w") as json_file:
    json_file.write(json_results)


if args.metrics:
    mm_reslts = np.array(mm_reslts)
    mm_gt = mm_gt[:np.where(mm_gt[:,0] == mm_reslts[:,0].max())[0].max()+1]

    def motMetricsEnhancedCalculator(gt, t):
        acc = mm.MOTAccumulator(auto_id=True)
        # Max frame number maybe different for gt and t files
        for frame in range(int(gt[:,0].max())):
            frame += 1 # detection and frame numbers begin at 1
            gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
            t_dets = t[t[:,0]==frame,1:6] # select all detections in t
            C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:])
            acc.update(gt_dets[:,0].astype('int').tolist(), t_dets[:,0].astype('int').tolist(), C)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['num_frames', 'recall', 'precision', 'num_switches', 'mota', 'motp'], name='acc')
        return summary


    print(motMetricsEnhancedCalculator(mm_gt, mm_reslts))




print(f'''Inference stats:
      \tFrames: {i}
      \tTime inference per frame: {(timedelta(seconds=stop-start).total_seconds()/i)*1000:.2f} msec
      \tFull inference time: {timedelta(seconds=stop-start)}''')
