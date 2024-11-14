import os
import cv2
import json
import numpy as np
from insightface.app import FaceAnalysis

def create_embedding_database(directory_path, output_file='embeddings.json'):
    # Initialize InsightFace model
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    embedding_db = []

    # Iterate through each image in the directory
    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        print(f"Processing image: {image_path}")
        # Analyze the current image
        faces = app.get(image)
        
        if len(faces) == 0:
            print(f"No face detected in image: {image_path}")
            continue

        # Extract the embedding of the first face detected
        face_embedding = faces[0].embedding.tolist()  # Convert to list for JSON serialization

        # Store the embedding with the image name
        embedding_db.append({
            'image_name': filename,
            'embedding': face_embedding
        })

    # Save the embeddings database to a JSON file
    with open(output_file, 'w') as f:
        json.dump(embedding_db, f)

    print(f"Embedding database created and saved to {output_file}")

def find_top_n_similar_faces_in_db(input_image_path, embedding_db_path='embeddings.json', top_n=5):
    # Initialize InsightFace model
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load and analyze the input image
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error loading image: {input_image_path}")
        return None

    input_faces = app.get(input_image)
    
    if len(input_faces) == 0:
        print("No face detected in the input image.")
        return None

    input_face_embedding = input_faces[0].embedding

    # Load the embeddings database from the JSON file
    with open(embedding_db_path, 'r') as f:
        embedding_db = json.load(f)

    # Calculate distances between input embedding and all embeddings in the database
    distances = []
    for entry in embedding_db:
        face_embedding = np.array(entry['embedding'])
        distance = np.linalg.norm(input_face_embedding - face_embedding)
        distances.append((entry['image_name'], distance))

    # Sort the distances and select the top N matches
    distances.sort(key=lambda x: x[1])
    top_n_matches = distances[:top_n]

    # Print the top N matches
    for i, (image_name, distance) in enumerate(top_n_matches, start=1):
        print(f"Rank {i}: {image_name} with a distance of {distance}")

    return top_n_matches

# Example usage:
directory_path = 'path/to/your/images/folder'
create_embedding_database(directory_path, output_file='embeddings.json')

input_image_path = 'path/to/your/input/image.jpg'
top_n_matches = find_top_n_similar_faces_in_db(input_image_path, embedding_db_path='embeddings.json', top_n=5)
