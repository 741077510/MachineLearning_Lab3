import numpy as np
import cv2
import os
from feature import NPDFeature
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#define a function to resize the image
def resize_image(dir):
    res_list = []
    for face_name in os.listdir(dir):
        img = cv2.imread(dir + '/' + face_name, 0)
        img_resize = cv2.resize(img, (24, 24), interpolation=cv2.INTER_AREA)
        res_list.append(img_resize)
    return res_list
def mk_dataset():
    face_dir = "/Users/zoushuai/Python/lab/datasets/original/face"
    nonface_dir = "/Users/zoushuai/Python/lab/datasets/original/nonface"
    face_list = resize_image(face_dir)
    nonface_list = resize_image(nonface_dir)
    train_set = face_list[0:250]
    train_set.extend(nonface_list[0:250])  # trainset contains 250 faces and 250 nonfaces
    train_set = np.array(train_set)
    validate_set = face_list[250:500]
    validate_set.extend(nonface_list[250:500])  # validateset contains 250 faces and 250 nonfaces
    validate_set = np.array(validate_set)
    train_img2feature_list = []
    validate_img2feature_list = []

    for i in range(500):
        npdFeature_train = NPDFeature(train_set[i])
        train_img2feature_list.append(npdFeature_train.extract())
        npdFeature_validate = NPDFeature(validate_set[i])
        validate_img2feature_list.append(npdFeature_validate.extract())
    train_img2feature_list = np.array(train_img2feature_list)
    validate_img2feature_list = np.array(validate_img2feature_list)
    AdaBoostClassifier.save(train_img2feature_list, 'train')
    AdaBoostClassifier.save(validate_img2feature_list, 'validate')

if __name__ == '__main__':
    mk_dataset()
    train_set_label = np.append(np.ones(250), (-1) * np.ones(250))
    validate_set_label = np.append(np.ones(250), (-1) * np.ones(250))
    train_img2feature_list = AdaBoostClassifier.load('train')
    validate_img2feature_list = AdaBoostClassifier.load('validate')

    adaboostClassifier = AdaBoostClassifier(DecisionTreeClassifier,20)
    classifier_list,alpha_list = adaboostClassifier.fit(train_img2feature_list,train_set_label)
    result_predict = adaboostClassifier.predict(validate_img2feature_list,classifier_list,alpha_list,0)

    f = open('report.txt','w')
    f.write(classification_report(validate_set_label, result_predict))


