import os
import cv2
import torch
import pandas as pd
import mediapipe as mp

from models.architecture import Architecture

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

os.environ["GLOG_minloglevel"] = "3"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Video batch generator
def video_batch_generator(df, batch_size=16):
    total_videos = len(df)
    for i in range(0, total_videos, batch_size):
        yield df.iloc[i:i + batch_size]

# Paths for loading datasets
datasets = {
    "./data/DAiSEE_Train.csv": "DAiSEE",
    "./data/EngageNet_Train.csv": "EngageNet"
}

FPS = 30
BATCH_SIZE = 16
LANDMARK_COUNT = 478
DIMENSION_COUNT = 3
EMBEDDING_COUNT = 128
CLASS_COUNT = 5

ITERATIONS = BATCH_SIZE * FPS

steps = ITERATIONS
batch_size = BATCH_SIZE
landmark_count = LANDMARK_COUNT
dimension_count = DIMENSION_COUNT
embedding_count = EMBEDDING_COUNT
class_count = CLASS_COUNT

for dataset_path, dataset_name in datasets.items():
    df = pd.read_csv(dataset_path)

    model = Architecture(landmark_count, dimension_count, embedding_count, class_count, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    # Process each video batch
    for i, video_batch in enumerate(video_batch_generator(df, batch_size=BATCH_SIZE)):
        print(f"\nBatch {i + 1} loading...")

        landmarks_batch = []
        labels_batch = []

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

                # Process frame for face landmarks with image dimensions
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extract 478 landmarks as a flat list (each landmark has x, y, z coordinates)
                        landmarks = [coord for lm in face_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                        
                        # Reshape the landmarks to (478, 3)
                        landmarks_array = torch.tensor(landmarks).reshape((478, 3)).float()

                        # Append landmarks and label to respective batches
                        landmarks_batch.append(landmarks_array)
                        labels_batch.append(torch.tensor([label], dtype=torch.int64))

                frame_index += 1

            cap.release()

        # Convert lists to tensors with appropriate shapes
        landmarks_batch = torch.stack(landmarks_batch)          # Shape: (B, 478, 3)
        labels_batch = torch.stack(labels_batch).squeeze()      # Shape: (B, 1)

        labels_batch = torch.nn.functional.one_hot(labels_batch, num_classes=class_count).long()

        for i in range(steps):
            optimizer.zero_grad()
            logits, loss = model(landmarks_batch.to(device), labels_batch.to(device))
            print(f"Iteration {i + 1:6d}, Loss: {loss.item():6f}")
            
            loss.backward()
            optimizer.step()

        # Optionally save or use the batches, e.g., store them in a file or use in a model.
        # torch.save(landmarks_batch, f"{dataset_name}_landmarks_batch_{i+1}.pt")
        # torch.save(labels_batch, f"{dataset_name}_labels_batch_{i+1}.pt")

    # Save the model after training on this dataset
    model_save_path = f"./models/{dataset_name}_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model for {dataset_name} saved to {model_save_path}")

# Release MediaPipe resources
face_mesh.close()
