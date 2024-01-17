import cv2
import numpy as np
import copy
import os
from .third_libs.OpenSeeFace.tracker import Tracker
#import imageio
from tqdm import tqdm
import sys
 
tar_size = 512

def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def get_length(pred):
    lm = np.array(pred)
    brow_avg = (lm[19] + lm[24]) * 0.5
    bottom = lm[8]
    length = distance(brow_avg, bottom)

    return length * 1.05

class OfflineReader:
    def __init__(self):
        '''
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num = 0
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        '''
        self.tracker = None

    def get_data(self, frame_rgb, frame_num):
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame_rgb
            
            if self.tracker is None:
                self.height, self.width = frame.shape[:2]
                
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                model_dir = os.path.join(cur_dir, 'third_libs/OpenSeeFace/models')
                
                self.tracker = Tracker(self.width, self.height, threshold=None, max_threads=1,
                              max_faces=1, discard_after=10, scan_every=30, 
                              silent=True, model_type=4, model_dir=model_dir, no_gaze=True, detection_threshold=0.6, 
                              use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=0)
                
            
            preds = self.tracker.predict(frame)
            if len(preds) == 0:
                print('No face detected in offline reader!')
                return False, False, []
            # try more times in the fisrt frame for better landmarks
            if frame_num == 0:
                for _ in range(3):
                    preds = self.tracker.predict(frame)
                    if len(preds) == 0:
                        print('No face detected in offline reader!')
                        return False, False, []
            lms = (preds[0].lms[:68, :2].copy() + 0.5).astype(np.int64)
            lms = lms[:, [1, 0]]
            
            
            #for (x, y) in lms:
            #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
            return True, frame, lms
      
        
