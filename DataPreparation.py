import pandas as pd
from glob import glob
import cv2
import numpy as np
import random

def get_id(file_path):
    return file_path.split('\\')[-1].split(".av")[0]

def get_value(row_entry):
    return row_entry.iloc[0].item()

class cell_study:
    def __init__(self, frame_rate, annotated_time):
        self.start_t = 31
        self.end_t = 43
        self.frame_rate = frame_rate
        if frame_rate == 3:
            self.inc = .33
        else:
            self.inc = .25
        self.stacked_time=[]
        self.t2 = annotated_time['t2']
        self.t3 = annotated_time['t3']
        self.t4 = annotated_time['t4']
        self.t5 = annotated_time['t5']

    def normalized_channel(self, x, type="cell"):
        if type == "time":
            return (x-self.start_t)/(self.end_t-self.start_t)
        else:
            return (x-2)/(5-2)

    def append_time(self, frame):
        if len(self.stacked_time) == 0:
            time_channel = np.full((128, 128),0.0, dtype=np.float16)
            self.stacked_time.append(self.start_t)
        else:
            time_value = self.stacked_time[-1]+self.inc
            if self.frame_rate == 3 and round(time_value % 1,2) == 0.99:
                time_value = round(time_value)
            self.stacked_time.append(time_value)
            norm_time = self.normalized_channel(time_value,"time")
            time_channel = np.full((128, 128), norm_time, dtype=np.float16)
        frame_with_time_channel = np.dstack([frame, time_channel])
        return frame_with_time_channel

    def append_cell_num(self,frame):
        time_value = self.stacked_time[-1]
        if time_value > self.t5:
            cell_num = 5
        elif time_value >= self.t4:
            cell_num = 4
        elif time_value >= self.t3:
            cell_num = 3
        else:
            cell_num = 2
        norm_cell_num = self.normalized_channel(cell_num)
        cell_channel = np.full((128, 128), norm_cell_num, dtype=np.float16)
        frame_with_cell_channel = np.dstack([frame, cell_channel])
        return frame_with_cell_channel


excel_path = "DataPrep/CellTest_Avoid.xlsx"
emb_vids_filePaths = glob('FramePred_Data/Cells/Avoid/*.avi')

video_meta_data= pd.read_excel(excel_path, index_col=None, header=0)
video_meta_data.fillna(100, inplace=True)

def get_data():
    data = []
    target = []
    for emb_vid_filePath in emb_vids_filePaths:
        video_id = get_id(emb_vid_filePath)
        meta_data = video_meta_data[video_meta_data['Well key'] == video_id]
        if not meta_data.empty:
            video = cv2.VideoCapture(emb_vid_filePath)
            start = get_value(meta_data['Frame_start'])
            end = get_value(meta_data['Frame_end'])
            annotated_time={
                            't2':get_value(meta_data['t2']),
                            't3':get_value(meta_data['t3']),
                            't4':get_value(meta_data['t4']),
                            't5':get_value(meta_data['t5'])
                            }
            frame_rate = get_value(meta_data['Frames per hour'])
            cellStage = cell_study(frame_rate,annotated_time)
            video_frames=[]
            while start < end:
                video.set(cv2.CAP_PROP_POS_FRAMES, start)
                ret, frame = video.read()
                if ret == False:
                    print("Didn't return a frame")
                    break
                frame = cv2.resize(frame, (128,128))
                normalized_frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                normalized_frame = normalized_frame.astype(np.float16)
                frame_time_channel = cellStage.append_time(normalized_frame)
                frame_cell_channel = cellStage.append_cell_num(frame_time_channel)
                video_frames.append(frame_cell_channel)
                start += 1
            frame_count = len(video_frames)
            i = 0
            while (i+7) < frame_count:
                data.append(video_frames[i:i+7])
                target.append(video_frames[i+7])
                i += 1

    data = np.array(data)
    target = np.array(target)

    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(indices))  # 80% train, 20% validation
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    train_x,train_y = data[train_indices], (target[train_indices], target[train_indices][..., :3])
    val_x, val_y = data[val_indices], (target[val_indices], target[val_indices][..., :3])

    return train_x, train_y, val_x, val_y


