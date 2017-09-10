# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from django.shortcuts import render
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import face_recognition
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
# EYE_AR_THRESH = 0.3
# EYE_AR_CONSEC_FRAMES = 3

### initialize the frame counters and the total number of blinks

COUNTER = 0
TOTAL = 0

# Initialize some variables for face_recognition
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# Load a sample picture and learn how to recognize it.
def load_sample_image_encoding(path):
    my_image = face_recognition.load_image_file(path)
    my_face_encoding = face_recognition.face_encodings(my_image)[0]
    return my_face_encoding

def recognise_faces(frame):
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces([my_face_encoding], face_encoding)
        name = "Unknown"

        if match[0]:
                name = "Vyom"
                
        face_names.append(name)
    return face_names

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def init_blink_detection():
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    shape_detector_file="shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(shape_detector_file)
    
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    return detector, predictor, lStart, lEnd, rStart, rEnd

def count_blinks(frame, EYE_AR_THRESH=0.3, EYE_AR_CONSEC_FRAMES=3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #INIT
    detector, predictor, lStart, lEnd, rStart, rEnd = init_blink_detection()
    
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #extract the left and right eye coordinates, then use the
        #coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            # reset the eye frame counter
            COUNTER = 0
        
        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, TOTAL
        

def start_video_stream(src=0):
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    # vs = FileVideoStream(args["video"]).start()
    fileStream = True
    # vs = VideoStream(src=0).start()
    vs = cv2.VideoCapture(src)
    return vs

def process_videostream(escape_key="q"):
    # loop over frames from the video stream
    while True:

        vs = start_video_stream(src=0)
        ret, frame = vs.read()
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        #get names of people recognised in the images
        # skipping every alternate frame for faster processing
        if process_this_frame:
            face_names = recognise_faces(frame)
        process_this_frame = not process_this_frame

        #count blinks
        frame, TOTAL = count_blinks(frame, EYE_AR_THRESH=0.3, EYE_AR_CONSEC_FRAMES=3)

        # show the frame
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord(escape_key):
            close_videostream(vs)
            break
        else:
            yeild (frame, face_names, TOTAL)

def close_videostream(videostream)
    print('Closing video stream...')
    videostream.release()
    cv2.destroyAllWindows()
    # fileStream = False
    time.sleep(1.0)
    
