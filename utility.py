import os
import cv2
import torch
import random

def load_subdirectories(path):
    sub_directories = []
    for subdir in os.listdir(path):
        if os.path.isdir(os.path.join(path, subdir)):
            subdir_path = os.path.join(path, subdir)
            sub_directories.append(subdir_path)
    sub_directories.sort()
    return sub_directories

def load_videos(path):
    videos = []
    for file in os.listdir(path):
        if file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(path, file)
            videos.append(video_path)
    videos.sort()
    return videos

def load_subject(path):
    videos = []
    subject_subdirectories = load_subdirectories(path)
    for subdirectory in subject_subdirectories:
        video_paths = load_videos(subdirectory)
        videos.extend(video_paths)
    videos.sort()
    return videos

def load_subjects(path, subject_count=8):
    subjects_subdirectories = load_subdirectories(path)
    subjects_subdirectories = random.sample(subjects_subdirectories, subject_count)

    videos = []
    for subject in subjects_subdirectories:
        subject_videos = load_subject(subject)
        videos.append(subject_videos)
    return videos

def get_frames(subject_videos, frame_interval=60, resize_to=None):
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
                frame_tensor = torch.tensor(frame, dtype=torch.float32)
                frames.append(frame_tensor)
        
        video_capture.release() 

        if frames:
            frames_subject.append(torch.stack(frames))
        else:
            print(f"No frames extracted from video file {subject_video}")

    if frames_subject:
        return torch.stack(frames_subject)
    else:
        return torch.tensor([])
    

parent_directory = 'DAiSEE/DataSet/Train/'
subdirectories = load_subdirectories(parent_directory)
subdirectory_name = random.sample(subdirectories, 1)[0]
subject_videos = load_subject(subdirectory_name)

frames_subject = get_frames(subject_videos)
print(frames_subject.shape)
