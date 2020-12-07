from deepface.extendedmodels import Emotion
from facenet_pytorch import MTCNN
from matplotlib import pyplot as plt
import argparse
import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from keras.preprocessing import image

class MTCNNFaceDetector():
    def __init__(self):
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        self.model = MTCNN(margin=20, select_largest=True, device=device)
        print(f">> Loaded MTCNN on {self.model.device}")

    def detect_faces(self, frame):
        """Returns faces bounding boxes"""
        tlbr_to_tlwh = lambda f: (f[0], f[1], f[2] - f[0], f[3] - f[1])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = Image.fromarray(rgb_frame)

        boxes, _ = self.model.detect(rgb_frame)
        if boxes is not None:
            coord = [tlbr_to_tlwh(box) for box in boxes.astype(int)][:1]
            
            return [max(0,c) for c in coord[0]]
        else:
            return []

def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Detect emotions in a video")
    parser.add_argument(
            "--output_folder",
            default="tmp_frames",
            type=str,
            help="the folder to output the dataset [default=tmp_frames]",
    )
    parser.add_argument(
            "--frame_limit",
            default=2**16,
            type=int,
            help="the number of frames to use [default=all]",
    )

    parser.add_argument(
            "--video_file",
            default="ur.mp4",
            type=str,
            help="the video file to generate the dataset from [default=ur.mp4]",
    )
    parser.add_argument(
            "--draw_boxes",
            action="store_true",
            help="whether or not to draw boxes on the frames",
    )
    parser.add_argument(
            "--scores_file",
            default="scores.csv",
            type=str,
            help="the location of the scores file [default=scores.csv]",
    )
    parser.add_argument(
            "--output_video",
            default="output.mp4",
            type=str,
            help="the location of the video [default=output.mp4]",
    )
    return parser.parse_args()

def collect_frames(video_file, frame_limit):
    """Collects frames from the video file"""
    frames = []
    i = 0
    cap = cv2.VideoCapture(video_file)
    while i < frame_limit and cap.isOpened():
        ret, frame = cap.read()
        i += 1
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def match_faces_bodies(frame, boxes, detector):
    """Matches faces with the bodies for the given frame"""
    canvas = frame.copy()
    df_faces = []
    for _, box in boxes.iterrows():
        roi = frame[int(box.y):int(box.y2), int(box.x):int(box.x2)]
        if roi.size > 0:
            faces = detector.detect_faces(roi)
            if faces:
                x, y, w, h = faces
                    
                df_faces.append({
                    'frame_id': int(box.abs_frame_id),
                    'box_id': int(box.box_id),
                    'x': int(box.x) + x,
                    'y': int(box.y) + y,
                    'x2': int(box.x) + x + w,
                    'y2': int(box.y) + y + h,
                })

    return canvas, df_faces


def detect_emotions(model, frame, draw_boxes):
    img, faces = frame
    df_faces = []
    for face in faces:
        face_image = img[face['y']:face['y2'], face['x']:face['x2']]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (48,48))
        #TO-DO: Align face
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis = 0)
        face_image /= 255 #normalize input in [0, 1]
        emotion = detect_emotion(model, face_image)
        # emotion = {'sad': 20}

        face.update({"emotion":emotion})
        df_faces.append(face)

    if draw_boxes:
        for face in df_faces:
            img = cv2.rectangle(
                    img,
                    (face['x'], face['y']),
                    (face['x2'], face['y2']),
                    (0, 255, 0), 
                    1
            )

            y = face['y']-2 if face['y']>10 else face['y2']
            
            img = cv2.putText(img,
                    str(face['emotion']).replace("'","")[1:-1], 
                    (face['x'], y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,
                    (255, 255,255),
                    1)
            
    return img, df_faces

def detect_emotion(model, img):
    resp_obj = {}
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotion_predictions = model.predict(img)[0,:]
    sum_of_predictions = emotion_predictions.sum()
    resp_obj["emotion"] = {}

    for i in range(0, len(emotion_labels)):
        emotion_label = emotion_labels[i]
        emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
        resp_obj["emotion"][emotion_label] = emotion_prediction

    return {emotion_labels[np.argmax(emotion_predictions)]:
            np.round(np.max(emotion_predictions),2)}


def main():
    args = parse_args()
    FRAME_LIMIT = args.frame_limit
    VIDEO_FILE = args.video_file
    DRAW_BOXES = args.draw_boxes
    SCORES_FILE = args.scores_file
    FOLDER = args.output_folder
    OUTPUT_VIDEO = args.output_video

    detector = MTCNNFaceDetector()
    model = Emotion.loadModel()

    df = pd.read_csv(SCORES_FILE, header=None, index_col=None, names=[
        "track_id", "frame_id", "box_id", "x", "y", "x2", "y2"] + list(range(80)))
    df["abs_frame_id"] = df.frame_id + df.track_id - 128
    df_faces = []
    print(f">> Reading video from {VIDEO_FILE}")
    frames = collect_frames(VIDEO_FILE, FRAME_LIMIT)

    print(">> Detecting emotions")
    for i, frame in tqdm(enumerate(frames), total=len(frames)):
        new_frame = match_faces_bodies(
                frame,
                df[df.abs_frame_id == i],
                detector=detector,
        )
        frames[i], new_df_faces = detect_emotions(model, new_frame, draw_boxes=DRAW_BOXES)

        df_faces += new_df_faces

    print(">> Creating video")
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height), True)
    for frame in tqdm(frames, total=len(frames)):
        out.write(frame)
    cv2.destroyAllWindows()
    out.release()

    df_faces = pd.DataFrame(df_faces)
    print(">> Saving box files")

    box_ids = df_faces.box_id.unique()
    for box_id in tqdm(box_ids, total=len(box_ids)):
        with open(os.path.join(FOLDER, f"person{box_id}.txt"), "w") as f:
            for _, frame in df_faces[df_faces.box_id == box_id].iterrows():
                bbox = str([frame.x, frame.y, frame.x2, frame.y2]).replace(" ", "").strip("[]")
                emotion = list(frame.emotion.keys())[0]
                f.write(f"{frame.frame_id},{bbox},{emotion},{frame.emotion[emotion]}\n")
    

if __name__ == "__main__":
    main()