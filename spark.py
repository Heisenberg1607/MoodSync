import sys
import os
import cv2
from mtcnn import MTCNN
from pyspark.sql import SparkSession
from deepface import DeepFace

# Get the batch number from the scheduler
batch_number = int(sys.argv[1])

# Initialize SparkSession
spark = SparkSession.builder.appName("ImageProcessing").getOrCreate()

# Initialize face detector
detector = MTCNN()

def process_image(image_path):
    """
    Function to process a single image:
    - Detect faces using MTCNN.
    - Use DeepFace for emotion recognition.
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return []

        # Detect faces
        faces = detector.detect_faces(image)

        emotions = []
        for face in faces:
            x, y, w, h = face['box']
            # Ensure bounding box is within image bounds
            x, y = max(0, x), max(0, y)

            # Crop the face
            face_crop = image[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            # Use DeepFace to analyze emotions
            try:
                analysis = DeepFace.analyze(face_crop, actions=["emotion"], enforce_detection=False)
                emotion_label = analysis['dominant_emotion']
                emotions.append(emotion_label)

                # Optional: Log analysis
                print(f"DeepFace Analysis for {image_path}: {analysis}")
            except Exception as e:
                print(f"DeepFace error for {image_path}: {e}")
                continue

        return emotions
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

# Locate the folder for the current batch
current_directory = os.path.dirname(os.path.abspath(__file__))
batch_folder = os.path.join(current_directory, f'MinuteBatch_{batch_number}')

# Read image paths from the batch folder
image_paths = [
    os.path.join(batch_folder, f)
    for f in os.listdir(batch_folder)
    if f.endswith('.png')
]

if not image_paths:
    print(f"No images found in {batch_folder}. Exiting.")
    spark.stop()
    exit()

# Parallelize image processing using Spark
rdd = spark.sparkContext.parallelize(image_paths)
results = rdd.map(process_image).collect()

# Aggregate results
emotion_counts = {}
for result in results:
    for emotion in result:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

# Determine the emotion with the highest count
if emotion_counts:
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    print(f"Dominant emotion in batch {batch_number}: {dominant_emotion} ({emotion_counts[dominant_emotion]} occurrences)")
else:
    print(f"No emotions detected in batch {batch_number}.")

# Stop SparkSession
spark.stop()
