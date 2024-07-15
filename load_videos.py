import os
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

parent_directory = 'DAiSEE/DataSet/Train'
videos = load_subjects(parent_directory, 1)
print(videos)