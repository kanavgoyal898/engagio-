import os
import pandas as pd

data_splits = ["Train", "Validation", "Test"]
video_directories = [f"../Engagement Datasets/EngageNet/EngageNet/{data_split}" for data_split in data_splits]
mapping_csv_path = f"../Engagement Datasets/EngageNet/EngageNet/&_engagement_labels.csv"
output_csv_path = f"./data/EngageNet_&.csv"
os.makedirs('./data/', exist_ok=True)

label_mapping = {
    'SNP(Subject Not Present)': 0,
    'Not-Engaged': 1,
    'Barely-engaged': 2,
    'Engaged': 3,
    'Highly-Engaged': 4
}

for video_directory in video_directories:
    data_split = video_directory.split('/')[-1]
    
    if not os.path.exists(video_directory):
        print(f"Directory {video_directory} does not exist.")
        continue

    video_files = [
        os.path.join(video_directory, file)
        for file in os.listdir(video_directory)
        if file.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    filenames = sorted([os.path.basename(file) for file in video_files])

    mapping_path_csv = mapping_csv_path.replace('&', data_split.lower())
    
    if not os.path.exists(mapping_path_csv):
        print(f"Error: Mapping CSV file {mapping_path_csv} does not exist.")
        continue

    mapping_df = pd.read_csv(mapping_path_csv)

    result_df = mapping_df[mapping_df['chunk'].isin(filenames)]

    result_df['Video Path'] = result_df['chunk'].apply(
        lambda filename: next((file for file in video_files if os.path.basename(file) == filename), None)
    )
    pd.set_option('future.no_silent_downcasting', True)
    result_df['label'] = result_df['label'].replace(label_mapping)

    final_df = result_df[['Video Path', 'chunk', 'label']].copy()
    final_df = final_df.rename(columns={'Video Path': 'video_path', 'chunk': 'chunk', 'label': 'label'})

    output_path_csv = output_csv_path.replace('&', data_split)
    final_df.to_csv(output_path_csv, index=False)

    print(final_df)
