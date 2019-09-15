from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import dlib
import fasent
from dis_calculate import extract_features, extract_features_from_img



def main():
    features = [[0, 0], [1, 1], [2, 2],[3,3],[4,4],[5,5],[6,6]]
    labels = [0, 1, 2,3,4,5,6]
    classifier = svm.SVC(gamma='scale')
    classifier.fit(features, labels)
    outputs = classifier.predict(features)

    report = classification_report(labels, outputs)
    print(report)

    
    out = classifier.predict([[0.5, 0.5]])
    print(out)


class FaceEmoPredictor():
    def __init__(self):
        pass

import cv2
detector = dlib.get_frontal_face_detector()  # Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Landmark identifier.
video = cv2.VideoCapture(0)
while True:
    succ, img = video.read()
    feature = extract_features_from_img(detector, predictor, img)
    out = classifier.predict([feature])
    print(out, id2label[out[0]])
    def predict(self, img):
        cv2.rectangle(img, (10,10), (100,100), (255,98,22), 1)
        return img
#return self.model.predict(img)
                          
