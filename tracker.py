import argparse
import cv2
import json
import mmcv
import torch
import numpy as np
import motmetrics as mm
from collections import defaultdict
from datetime import timedelta
from time import time
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import ops


def check_path(path):
        if Path.exists(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

class Tracking():
    def __init__(self, tracker_type, target_video, model_weights, first_frame,
                 last_frame=None, gt_path=None, make_out_video=False, output_video_path=None, show_video=False):
        self.mm_results = []
        self.empty_tracks = set()
        self.track_history = defaultdict(lambda: defaultdict(list))
        self.metrics = False
        if gt_path is not None:
            self.gt = np.loadtxt(gt_path, delimiter=',')
            self.metrics = True
        self.make_out_video = make_out_video
        self.show_video = show_video
        self.tracker_type = tracker_type
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.iter_times = []

        self.detect_model = YOLO(model_weights)
        self.cap = mmcv.VideoReader(target_video)
        if self.make_out_video:
            self.out = cv2.VideoWriter(output_video_path,
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  self.cap.fps,
                                  (self.cap.width, self.cap.height))

        if last_frame is not None:
            self.cap = self.cap[first_frame-1:last_frame]
        else:
            self.cap = self.cap[first_frame-1:]


        if tracker_type == 'deep':
            from deep_sort_realtime.deepsort_tracker import DeepSort
            tracker_args = {
                'max_age': 1,
                'max_iou_distance': 0.8,
                'max_cosine_distance': 0.1,
                'n_init': 6
            }
            self.tracker = DeepSort(**tracker_args)

        elif tracker_type =='smile':
            from ultralytics.utils import IterableSimpleNamespace
            from smile_note import SMILEtrack

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
                'with_reid': True
                })
            self.tracker = SMILEtrack(tracker_args)

    def prepare_det_res(self, res):
        '''Formats detector's output to proper trackrer input
        since different trackers have different formats of input
        input: det_results; output: formated det_results'''
        if self.tracker_type == 'deep':
            return [[*ops.xyxy2ltwh(res[0].boxes[i].xyxy.cpu()).tolist(),
                     res[0].boxes[i].conf.cpu().item(),
                     res[0].boxes[i].cls.cpu().item()] for i in range(len(res[0].boxes))]

    # Following only needed if using the built-in smiletrack
        # elif self.tracker_type == 'smile':
        #     return [res[0].boxes.xyxy.cpu(),
        #             res[0].boxes.id.int().cpu().tolist(),
        #             res[0].boxes.cls.cpu(),
        #             res[0].boxes.conf.cpu()]

    def populate_history(self, frame_i, x1, y1, x2, y2, conf, cls, track_id):
        '''Creating a JSON-like structure "track_history" with
        xyxy box coordinates, class_ids and center coodinates for each track_id'''
        if self.metrics:
            self.mm_results.append([self.first_frame+frame_i, track_id, x1, y1, x2-x1, y2-y1, -1, int(cls), conf])
        history = self.track_history[int(track_id)]
        history['coord'].append([float(i) for i in [x1, y1, x2, y2]])
        history['class_id'].append(int(cls))
        history['center'].append([float(x1+x2)/2, float(y1+y2)/2])
        if len(history['coord']) > 30:
            history['coord'].pop(0)
            history['class_id'].pop(0)
            history['center'].pop(0)

    def plot_tracks(self, track_id, frame):
        '''Plots a track path on the frame if make_out_video==True
        input: frame; output: frame'''
        points = np.hstack(self.track_history[track_id]['center']).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=4)
        return frame


    def process_frame(self, frame_i, frame):
        '''Processes a frame depending on the "tracker_type":
        detection -> tracking -> writing history and making output video with tracks printed;
        measures detection an tracking time on the per frame basis
        input: frame; output: frame'''
        if self.tracker_type == 'smile':
            start = time()
            # det_results = self.detect_model.track(frame, persist=True, tracker="smiletrack.yaml")
            det_results = self.detect_model(frame)
            tr_results = self.tracker.update(det_results[0].boxes, frame)
            stop = time()
            if self.make_out_video:
                frame = det_results[0].plot()
            # for box, track_id, cls, conf in zip(*self.prepare_det_res(det_results)): # only for the built-in smiletrack
            for x1, y1, x2, y2, track_id, conf, cls, _ in tr_results:
                self.populate_history(frame_i, x1, y1, x2, y2, conf, cls, track_id)
                if self.make_out_video:
                    self.plot_tracks(track_id, frame)

        elif self.tracker_type == 'deep':
            start = time()
            det_results = self.detect_model(frame)
            tr_results = self.tracker.update_tracks(self.prepare_det_res(det_results), frame=frame)
            stop = time()
            if self.make_out_video:
                frame = det_results[0].plot()
            for row in tr_results:
                x1, y1, x2, y2 = row.to_ltrb()
                cls = int(row.get_det_class())
                conf = row.get_det_conf()
                track_id = int(row.track_id)
                self.populate_history(frame_i, x1, y1, x2, y2, conf, cls, track_id)
                if self.make_out_video:
                    self.plot_tracks(track_id, frame)

        self.iter_times.append(timedelta(seconds=stop-start))
        return frame

    def track(self):
        '''Iterates over frames with exception handling if the IndexError is encountered
        IndexError means that all the tracks are empty, in this case we just continue iterating
        writing an empty frame in output video (if make_out_video==True)'''
        for frame_i, frame in enumerate(self.cap):
            try:
                frame = self.process_frame(frame_i, frame)
                if self.make_out_video:
                    self.write_frame(frame_i, frame)

                if self.show_video:
                    cv2.namedWindow('video', 0)
                    cv2.imshow(frame, 'video', 1)

            except IndexError:
                self.empty_tracks.add(self.first_frame+frame_i)
                if self.make_out_video:
                    self.write_frame(frame_i, frame, empty_tr=True)
                print(f'Empty track detected on the frame {self.first_frame+frame_i}')
                continue

    def write_frame(self, frame_i, frame, empty_tr=False):
        '''Writes a frame to self.out with a text overlay (frame number) in the left corner'''
        if empty_tr:
            text = f'{self.first_frame+frame_i} EMPTY TRACK'
            color = (30, 30, 255)
        else:
            text = f'{self.first_frame+frame_i}'
            color = (0, 255, 255)

        cv2.putText(frame,
                    text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX , 2,
                    color,
                    2,
                    cv2.LINE_4)
        self.out.write(frame)

    def history_to_json(self):
        '''Dumps "track_history" to disk as a JSON file'''
        json_results = json.dumps(self.track_history)
        with open("./tracking_results.json", "w") as json_file:
            json_file.write(json_results)

    def mot_metrics(self):
        '''Computes and returns metrics MOTA, MOTP, Precision, Recall and the number of switches
        This is a modified version of the script one can found here: https://github.com/cheind/py-motmetrics
        as a motMetricsEnhancedCalculator function'''
        t = np.array(self.mm_results)
        gt = self.gt[np.where(self.gt[:,0] == t[:,0].min())[0].min():
                     np.where(self.gt[:,0] == t[:,0].max())[0].max()+1] #slicing gt at max and min frame numbers found in t
        acc = mm.MOTAccumulator(auto_id=True)
        for frame in range(int(gt[:,0].max())):
            frame += 1 # detection and frame ndef check_path(path):
            gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
            t_dets = t[t[:,0]==frame,1:6] # select all detections in t
            C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:])
            acc.update(gt_dets[:,0].astype('int').tolist(), t_dets[:,0].astype('int').tolist(), C)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['num_frames', 'recall', 'precision', 'num_switches', 'mota', 'motp'], name='acc')
        return summary

    def print_results(self):
        if self.metrics:
            print(f'\n {self.mot_metrics()} \n')

        if len(self.empty_tracks) > 0:
            print('Empty track found on frames:')
            print(self.empty_tracks, '\n')

        print(f'''Inference stats:
              \tFrames: {len(self.cap)}
              \tTime inference per frame: {(sum(self.iter_times, timedelta(0)) / len(self.iter_times)).total_seconds()*1000:.2f} msec
              \tSTD of inference time: {np.array([x.total_seconds()*1000 for x in self.iter_times]).std():.2f} msec
              \tFull inference time: {sum(self.iter_times, timedelta(0))} sec''')

if __name__ == "__main__":
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Tracker options', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt_path', type=check_path, default=Path('./Datasets/gt/gt.txt'), help='File with ground truth labels')
    parser.add_argument('--imgs_path', type=check_path, default=Path('./Datasets/v2/images/'), help='Input dir for images')
    parser.add_argument('--videos_path', type=check_path, default=Path('./Videos/'), help='Input dir for videos')
    parser.add_argument('--input_video_path', type=str, default=None, help='Input path for new video')
    parser.add_argument('--output_video_path', type=str, default=Path('./output.mp4'), help='Output path for a processed video')
    parser.add_argument('--model_weights', type=check_path, default=Path('./Models/ultralytics/yolov10x_v2_4_best.pt'), help='YOLO model weights path')
    parser.add_argument('--tracker', type=str, default='smile', choices=['smile', 'deep'], help='Tracker model: ["smile", "deep"]')
    parser.add_argument('--show_video', action=argparse.BooleanOptionalAction, help='Wheither to show video of tracking or not')
    parser.add_argument('--metrics', action=argparse.BooleanOptionalAction, help='Compute metrics or not')
    parser.add_argument('--make_output_video', action=argparse.BooleanOptionalAction, help='Form a video from tracker`s output')
    parser.add_argument('--first_frame', type=int, default=1, help='Start processing at this frame')
    parser.add_argument('--last_frame', type=int, default=None, help='Finish processing at this frame')
    args = parser.parse_args()

    video = str(args.videos_path / '31-03-2024-09%3A34%3A24.mp4')
    if args.input_video_path is not None:
        video = args.input_video_path

    gt_path = None
    if args.metrics:
        gt_path = args.gt_path

    tr_model = Tracking(args.tracker,
                        video,
                        args.model_weights,
                        args.first_frame,
                        last_frame=args.last_frame,
                        make_out_video=args.make_output_video,
                        gt_path=gt_path,
                        output_video_path = args.output_video_path,
                        show_video=args.show_video
                        )
    tr_model.track()
    tr_model.print_results()
    tr_model.history_to_json()
