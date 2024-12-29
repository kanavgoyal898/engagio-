import os
import csv
import cv2
import pandas as pd
import mediapipe as mp

os.environ["GLOG_minloglevel"] = "3"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Video batch generator
def video_batch_generator(df, batch_size=16):
    total_videos = len(df)
    for i in range(0, total_videos, batch_size):
        yield df.iloc[i:i + batch_size]

# Paths for saving landmarks
output_files = {
    "./data/DAiSEE_Train.csv": "./data/DAiSEE_Landmarks.csv",
    "./data/EngageNet_Train.csv": "./data/EngageNet_Landmarks.csv"
}

BATCH_SIZE = 16

for dataset, output_csv in output_files.items():
    df = pd.read_csv(dataset)

    # Initialize CSV file with headers (open file once)
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        headers = ["video", "frame", "chunk", "label"] + [f"landmark_{i}_{axis}" for i in range(478) for axis in ["x", "y", "z"]]
        writer.writerow(headers)

        for i, video_batch in enumerate(video_batch_generator(df, batch_size=BATCH_SIZE)):
            print(f"\nBatch {i + 1} loading...")

            for _, row in video_batch.iterrows():
                video = row['video_path']
                chunk = row.get('chunk', 0)
                label = row.get('label', 0)

                cap = cv2.VideoCapture(video)
                frame_index = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR frame to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Get image dimensions (width and height)
                    height, width, _ = frame.shape

                    square_dimension = min(height, width)
                    square_frame = cv2.resize(rgb_frame, (square_dimension, square_dimension))

                    # Process frame for face landmarks with image dimensions
                    results = face_mesh.process(square_frame)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            # Extract 478 landmarks as a flat list
                            landmarks = [coord for lm in face_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                            
                            # Write to CSV (append the current row)
                            writer.writerow([video, frame_index, chunk, label] + landmarks)

                    frame_index += 1

                cap.release()

            print(f"Batch {i + 1} processed successfully.")

    print(f"\n{dataset} processed and saved to {output_csv}.")

# Release MediaPipe resources
face_mesh.close()
