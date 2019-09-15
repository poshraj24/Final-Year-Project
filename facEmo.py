import math

import cv2
import dlib
import numpy as np
from sklearn.externals import joblib


def distance(shape, pnt_1, pnt_2):
    a = (shape.part(pnt_2).x - shape.part(pnt_1).x) ** 2
    b = (shape.part(pnt_2).y - shape.part(pnt_1).y) ** 2
    dis = math.sqrt((a + b))
    return dis


#

def detect_faces(detector, img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = detector(gray, 1)  # Detect the faces in the image
        if len(detections) == 0:
            return None
        return detections
    except Exception as e:
        print(e)
        return None


def extract_features_from_img(predictor, faces, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    for i, face in enumerate(faces):  # For all detected face instances individually
        try:
            shape = predictor(gray, face)  # Draw Facial Landmarks with the predictor class
            # Calculate average distance between right brows and  right eye
            dist_1937 = distance(shape, 19, 37)
            dist_2038 = distance(shape, 20, 38)
            FP01 = (dist_1937 + dist_2038) / 2

            # Calculate average distance between left brows and left eye
            dist_2343 = distance(shape, 23, 43)
            dist_2444 = distance(shape, 24, 44)
            FP02 = (dist_2343 + dist_2444) / 2

            FP03 = distance(shape, 21, 39)  # Calculate distance between left corner point of right eye and brows
            FP04 = distance(shape, 17, 36)  # Calculate distance between right corner point of right eye and brows
            FP05 = distance(shape, 22, 42)  # Calculate distance between right corner point of left eye and brows
            FP06 = distance(shape, 26, 45)  # Calculate distance between left corner point of left eye and brows
            FP07 = distance(shape, 21, 22)  # Calculate distance between corner point of two eyes

            FP08 = distance(shape, 27,
                            31)  # Calculate distance between upper nose point and right most point of lower nose
            FP09 = distance(shape, 27,
                            35)  # Calculate distance between upper nose point and left most point of lower nose
            FP10 = (FP08 + FP09) / 2
            FP11 = distance(shape, 30,
                            33)  # Calculate distance between lower centre nose point and upper centre nose point.
            dist_3150 = distance(shape, 31,
                                 50)  # calculate distance between nose right corner and right corner of upper lips
            dist_3552 = distance(shape, 35,
                                 52)  # calculate distance between nose left corner and left corner of upper lips
            FP12 = (dist_3150 + dist_3552) / 2
            dist_3148 = distance(shape, 31, 48)
            dist_3554 = distance(shape, 35, 54)
            FP12 = (dist_3148 + dist_3554) / 2
            FP13 = distance(shape, 48, 54)
            FP14 = distance(shape, 61, 67)  # calculate distance between right corner of inner lips
            FP15 = distance(shape, 62, 66)  # calculate distance between left corner of inner lips
            FP16 = distance(shape, 63, 65)  # calculate distance between middle of inner lips
            FP17 = distance(shape, 58, 7)  # calculate distance between right corner of lower lips and right chin
            FP18 = distance(shape, 57, 8)  # calculate distance between middle of lower lips and middle chin
            FP19 = distance(shape, 56, 9)  # calculate distance between left corner of lower lips and left chin
            FP20 = distance(shape, 60, 64)  # calculate distance between inner lips corner
            feature = [FP01, FP02, FP03, FP04, FP05, FP06, FP07, FP08, FP09, FP10,
                       FP11, FP12, FP13, FP14, FP15, FP16, FP17, FP18, FP19, FP20]
            features.append(np.asarray(feature, dtype=np.float32))
        except Exception as e:
            print(e)
            features.append(None)
    return features if len(features) > 0 else None


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()  # Face detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Landmark identifier.
    video = cv2.VideoCapture(0)
    # classifier = svm.SVC()
    classifier = joblib.load('facEmo_saved_model2.pkl')
    id2label = {0: "ANGRY", 1: "DISGUST", 2: "FEAR", 3: "HAPPY", 4: "NEUTRAL", 5: "SAD", 6: "SURPRISE"}
    while True:
        succ, img = video.read()
        if not succ:
            continue
        faces = detect_faces(detector, img)
        if faces is not None:
            features = extract_features_from_img(predictor, faces, img)
            if features is None:
                continue
            for face, feature in zip(faces, features):
                if feature is None:
                    continue
                print("feature: {}, shape: {}".format(feature, feature.shape))
                feature = feature.reshape(-1, len(feature))
                out = classifier.predict(feature)
                label = id2label[out[0]]
                print("feature: {}, shape: {}, face: {}, sentiment: {}".format(feature, feature.shape, face, label))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (255, 94, 94), 2)
                cv2.putText(img, label, (face.left(), face.top()), font, 1, (255, 255, 255), 1)
        cv2.imshow("FacEmo", img)  # Display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit program when the user presses 'q'
            break

'''from sklearn import svm
from sklearn.metrics import classification_report

if __name__ == '__main__':
    features = [[0, 0], [1, 1], [2, 2]]
    labels = [0, 1, 2]
    classifier = svm.SVC(gamma='scale')
    classifier.fit(features, labels)
    outputs = classifier.predict(features)

    report = classification_report(labels, outputs)
    print(report)

    out = classifier.predict([[0.5, 0.5]])
    print(out)
    '''
