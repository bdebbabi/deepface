from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.preprocessing import image
from facenet_pytorch import MTCNN
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import shutil
import torch
import wget
import cv2
import os


class MTCNNFaceDetector():
    def __init__(self):
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        self.model = MTCNN(margin=20, select_largest=True, device=device)
        print(f">> Loaded MTCNN on {self.model.device}")

    def detect_faces(self, frame, threshold):
        """Returns faces bounding boxes"""
        tlbr_to_tlwh = lambda f: (f[0], f[1], f[2] - f[0], f[3] - f[1])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = Image.fromarray(rgb_frame)

        boxes, probabilities = self.model.detect(rgb_frame)
        boxes = [box for box, prob in zip(boxes.astype(int), probabilities) if prob >= threshold/100]
        if boxes is not None:
            coord = [tlbr_to_tlwh(box) for box in boxes]
            return coord
        else:
            return []

class HaarFaceDetector():
    def __init__(self):
        cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

        self.model = cv2.CascadeClassifier(haar_model)
        print(f">> Loaded Haar face detector")

    def detect_faces(self, frame, threshold):
        """Returns faces bounding boxes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.model.detectMultiScale(gray, 1.3, 5)

        return faces 

def load_face_detector(face_detector):
    if face_detector == 'haar':
        return HaarFaceDetector()
    elif face_detector == 'cnn':
        return MTCNNFaceDetector()
    else:
        raise Exception(f"detection method {face_detector} is invalid, expected one of [haar, cnn]")

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

def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Detect emotions in a video")
    parser.add_argument(
            "--output_folder",
            default="tmp_frames",
            type=str,
            help="the folder to output the tracking predictions [default=tmp_frames]",
    )
    parser.add_argument(
            "--frame_limit",
            default=2**16,
            type=int,
            help="the number of frames to use [default=all]",
    )
    parser.add_argument(
            "--use_webcam",
            action="store_true",
            help="whether or not to use the webcam instead of the input video [default=False]",
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
            help="whether or not to draw boxes on the frames [default=False]",
    )
    parser.add_argument(
            "--output_video",
            default="output.mp4",
            type=str,
            help="the location of the video [default=output.mp4]",
    )
    parser.add_argument(
            "--face_detector",
            default="haar",
            type=str,
            help="face detector to use: haar (faster) or mtcnn (more precise) [default=haar]",
    )
    parser.add_argument(
            "--scores_file",
            default="scores.csv",
            type=str,
            help="the location of the scores file [default=scores.csv]",
    )
    parser.add_argument(
            "--track_boxes",
            action="store_true",
            help="whether or not to track boxes on the frames [default=False]",
    )
    parser.add_argument(
            "--threshold",
            default=98,
            type=int,
            help="face detection threshold for mtcnn [default=98]",
    )
    return parser.parse_args()



def load_model():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    num_classes = 7

    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    #2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    #3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    model.add(Flatten())

    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    weights_url = 'https://github.com/serengil/tensorflow-101/raw/master/model/facial_expression_model_weights.h5'
    weights_path = os.path.basename(weights_url)    
    if not os.path.exists(weights_path):
        wget.download(weights_url)

    model.load_weights(weights_path) #load weights

    return model

def match_faces_bodies(frame, boxes, detector, threshold):
    """Matches faces with the bodies for the given frame"""
    df_faces = []
    for _, box in boxes.iterrows():
        roi = frame[int(box.y):int(box.y2), int(box.x):int(box.x2)]
        if roi.size > 0:
            faces = detector.detect_faces(roi, threshold)
            if faces:
                x, y, w, h = faces[0]
                    
                df_faces.append({
                    'frame_id': int(box.abs_frame_id),
                    'box_id': int(box.box_id),
                    'x': int(box.x) + x,
                    'y': int(box.y) + y,
                    'x2': int(box.x) + x + w,
                    'y2': int(box.y) + y + h,
                })

    return df_faces

def detect_emotions(model, img, faces, draw_boxes, track_boxes):
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    df_faces = []
    out_img = img.copy()
    for face in faces:
        if track_boxes:
            x, y, w, h = face['x'], face['y'], face['x2']-face['x'], face['y2']-face['y']
        else:
            x, y, w, h = face
        out_img = cv2.rectangle(out_img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
        
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
        
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
        
        #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
        
        emotion = emotions[max_index]
        score = str(np.round(predictions[0][max_index],2))
        #write emotion text above rectangle
        out_img = cv2.putText(out_img, f'{emotion}: {score}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        #process on detected face end
        #-------------------------
        if track_boxes:
            face.update({"emotion":{emotion:score}})
            df_faces.append(face)

    return out_img, df_faces

#-----------------------------
def main():
    args = parse_args()
    FRAME_LIMIT = args.frame_limit
    VIDEO_FILE = args.video_file
    DRAW_BOXES = args.draw_boxes
    SCORES_FILE = args.scores_file
    FOLDER = args.output_folder
    OUTPUT_VIDEO = args.output_video
    FACE_DETECTOR = args.face_detector
    USE_WEBCAM = args.use_webcam
    TRACK_BOXES = args.track_boxes
    THRESHOLD = args.threshold

    detector = load_face_detector(FACE_DETECTOR)
    
    model = load_model()

    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
        while(True):
            ret, img = cap.read()

            faces = detector.detect_faces(img, THRESHOLD)
            img, _ = detect_emotions(model, img, faces, DRAW_BOXES, TRACK_BOXES)

            cv2.imshow('img',img)

            if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
                break

        #kill open cv things		
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print(f">> Reading video from {VIDEO_FILE}")
        frames = collect_frames(VIDEO_FILE, FRAME_LIMIT)

        if TRACK_BOXES:
            df = pd.read_csv(SCORES_FILE, header=None, index_col=None, names=[
            "track_id", "frame_id", "box_id", "x", "y", "x2", "y2"] + list(range(80)))
            df["abs_frame_id"] = df.frame_id + df.track_id - 128
            df_faces = []

            print(">> Detecting emotions")
            for i, frame in tqdm(enumerate(frames), total=len(frames)):
                faces = match_faces_bodies(
                        frame,
                        df[df.abs_frame_id == i],
                        detector=detector,
                        threshold=THRESHOLD
                )
                frames[i], new_df_faces = detect_emotions(model, frame, faces, DRAW_BOXES, TRACK_BOXES)

                df_faces += new_df_faces

            df_faces = pd.DataFrame(df_faces)
            print(">> Saving box files")

            if os.path.exists(FOLDER):
                shutil.rmtree(FOLDER)
            os.mkdir(FOLDER)
            box_ids = df_faces.box_id.unique()
            for box_id in tqdm(box_ids, total=len(box_ids)):
                with open(os.path.join(FOLDER, f"person{box_id}.txt"), "w") as f:
                    for _, frame in df_faces[df_faces.box_id == box_id].iterrows():
                        bbox = str([frame.x, frame.y, frame.x2, frame.y2]).replace(" ", "").strip("[]")
                        emotion = list(frame.emotion.keys())[0]
                        f.write(f"{frame.frame_id},{bbox},{emotion},{frame.emotion[emotion]}\n")

        else:
            print(">> Detecting emotions")
            for i, frame in tqdm(enumerate(frames), total=len(frames)):
                faces = detector.detect_faces(frame, THRESHOLD)
                frames[i], _ = detect_emotions(model, frame, faces, DRAW_BOXES, TRACK_BOXES)

        print(">> Creating video")
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height), True)
        for frame in tqdm(frames, total=len(frames)):
            out.write(frame)
        cv2.destroyAllWindows()
        out.release()

if __name__ == "__main__":
    main()