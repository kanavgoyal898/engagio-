import os
import pandas as pd

data_splits = ["Train", "Validation", "Test"]
video_directories = [f"../Engagement Datasets/DAiSEE/DAiSEE/DataSet/{data_split}" for data_split in data_splits]
mapping_csv_path = f"../Engagement Datasets/DAiSEE/DAiSEE/Labels/&Labels.csv"
output_csv_path = f"./data/DAiSEE_&.csv"
os.makedirs('./data/', exist_ok=True)

for video_directory in video_directories:
    data_split = video_directory.split('/')[-1]
    
    if not os.path.exists(video_directory):
        print(f"Error: Directory {video_directory} does not exist.")
        continue

    video_files = []
    for root, dirs, files in os.walk(video_directory):
        level_depth = root.replace(video_directory, '').count(os.sep)
        if level_depth == 2:
            video_files.extend([
                os.path.join(root, file)
                for file in files
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv'))
            ])

    filenames = sorted([os.path.basename(file) for file in video_files])

    mapping_path_csv = mapping_csv_path.replace('&', data_split)
    
    if not os.path.exists(mapping_path_csv):
        print(f"Error: Mapping CSV file {mapping_path_csv} does not exist.")
        continue

    mapping_df = pd.read_csv(mapping_path_csv)

    result_df = mapping_df[mapping_df['ClipID'].isin(filenames)]

    result_df['Video Path'] = result_df['ClipID'].apply(
        lambda filename: next((file for file in video_files if os.path.basename(file) == filename), None)
    )

    final_df = result_df[['Video Path', 'ClipID', 'Engagement']].copy()
    final_df = final_df.rename(columns={'Video Path': 'video_path', 'ClipID': 'chunk', 'Engagement': 'label'})

    output_path_csv = output_csv_path.replace('&', data_split)
    final_df.to_csv(output_path_csv, index=False)

    print(final_df)
