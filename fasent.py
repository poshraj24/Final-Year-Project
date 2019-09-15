import dlib
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.externals import joblib

from dis_calculate import extract_features, extract_features_from_img
from utils import create_dataset

if __name__ == '__main__':
    data_dir = "C:/Users\shish\Desktop\FacEmo\COHN_KANADE"
    train_images, train_labels, test_images, test_labels, id2label = \
        create_dataset(data_dir, 0.2, hot_labels=False, max_classes=7)
    print("Train")
    print(train_images, train_images.shape)
    print(train_labels, train_labels.shape)
    print("Test")
    print(test_images, test_images.shape)
    print(test_labels, test_labels.shape)
    print("Id2label")
    print(id2label)

    train_features = []
    train_y = []

    test_features = []
    test_y = []

    detector = dlib.get_frontal_face_detector()  # Face detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Landmark identifier.

    # num_cores = multiprocessing.cpu_count()
    # train_features = Parallel(n_jobs=num_cores)(delayed(extract_features)(detector, predictor, image) for image in train_images)
    # print(train_features)

    for i, image in enumerate(train_images):
        if i % 20 == 0:
            print("Completed " + str(i) + " train samples")
        try:
            feature = extract_features(detector, predictor, image)
            if feature is not None:
                train_features.append(feature)
                train_y.append(train_labels[i])
        except Exception as e:
            print(e)

    for i, image in enumerate(test_images):
        if i % 20 == 0:
            print("Completed " + str(i) + "test samples")
        try:
            feature = extract_features(detector, predictor, image)
            if feature is not None:
                test_features.append(feature)
                test_y.append(test_labels[i])
        except Exception as e:
            print(e)

    train_features = np.array(train_features)
    test_features = np.array(test_features)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    print("Train features")
    print(train_features, train_features.shape)
    print(train_y, train_y.shape)
    print("Test features")
    print(test_features, test_features.shape)
    print(test_y, test_y.shape)

    classifier = svm.SVC(kernel = 'linear', C= 1)
    classifier.fit(train_features, train_y)
    joblib.dump(classifier, 'facEmo_saved_model2.pkl')

    train_outputs = classifier.predict(train_features)
    test_outputs = classifier.predict(test_features)
    test_report = classification_report(test_y, test_outputs)
    train_report = classification_report(train_y,train_outputs)
    print("Training result is : " + train_report)
    print("Testing result is : " + test_report)


