import os
import random
import cv2
import torch
import torch.nn as nn
import pandas as pd

PATH = 'DAiSEE/DataSet/'

FPS = 30
BATCH_SIZE = 8
VIDEO_LENGTH = 10
FRAME_INTERVAL = 2

def get_subdirectories(path, sort=False):
    """Get a list of sub-directories in parent directory 'path'."""
    sub_directories = [os.path.join(path, subdir) for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir))]
    
    if sort:
        sub_directories.sort()
    else:
        random.shuffle(sub_directories)
    
    return sub_directories

def get_videos(path, sort=False):
    """Get a list of video paths from a directory 'path'."""
    videos = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(('.mp4', '.avi', '.mov'))]
    
    if sort:
        videos.sort()
    else:
        random.shuffle(videos)
    
    return videos

def get_video_paths(path, sort=False):
    """Get all video paths in a dataset."""
    videos = []
    subject_list = get_subdirectories(path)
    
    for subject in subject_list:
        subject_subdir_list = get_subdirectories(subject, sort)
        
        for subdir in subject_subdir_list:
            subdir_video_paths = get_videos(subdir, sort)
            videos.extend(subdir_video_paths)
    
    return videos

def get_frames(subject_videos, frame_interval=FRAME_INTERVAL, resize_to=None):
    """Get frames from a list of 'subject_videos' paths."""
    frames_subject = []

    for subject_video in subject_videos:
        video_capture = cv2.VideoCapture(subject_video)
        
        if not video_capture.isOpened():
            print(f"Error opening video file {subject_video}")
            continue

        count = 0
        frames = []

        while True:
            success, frame = video_capture.read()
            if not success:
                break

            count += 1
            if count % frame_interval == 0:
                if resize_to:
                    frame = cv2.resize(frame, resize_to)

                # Convert the first frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame_tensor = torch.from_numpy(frame)
                frames.append(frame_tensor)

        video_capture.release()

        if frames:
            frames_required = (FPS * VIDEO_LENGTH) // frame_interval
            frames = frames[:frames_required]
            while len(frames) < frames_required:
                frames.append(frames[-1])
            frames_subject.append(torch.stack(frames))
        else:
            print(f"No frames extracted from video file {subject_video}")

    return torch.stack(frames_subject) if frames_subject else torch.tensor([])

def get_labels(paths):
    """Get labels for boredom, engagement, confusion, frustration."""
    data = pd.read_csv('DAiSEE/Labels/AllLabels.csv')
    tails = [os.path.split(path)[1] for path in paths]
    filtered_data = data[data['ClipID'].isin(tails)]
    engagement_data = filtered_data[['Engagement']]
    return torch.tensor(engagement_data.values)

def load_data(path, dataset, batch_size):
    """Load random videos from 'path'."""
    path = os.path.join(path, dataset)
    paths = get_video_paths(path)
    random_paths = random.sample(paths, batch_size)
    X = get_frames(random_paths, FRAME_INTERVAL)
    Y = get_labels(random_paths)
    return X, Y