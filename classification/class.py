import joblib
import numpy as np
import argparse
import os
import cv2
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def get_labels(tr_img_list, te_img_list, num_classes):
    temp = {}
    for i in tr_img_list:
        img = i[:-7]
        if img not in temp:
            temp[img] = 1
        elif img in temp:
            temp[img] += 1
    for i in te_img_list:
        img = i[:-7]
        if img not in temp:
            temp[img] = 1
        elif img in temp:
            temp[img] += 1

    li = []
    for i in temp:
        if i not in li and temp[i] >= 60:
            li.append(i)

    lbl_dict = {}
    for i in range(len(num_classes)):
        # lbl_dict[li[num_classes[i]]] = i for reset the label
        lbl_dict[li[num_classes[i]]] = num_classes[i]

    tr_data_li = []
    for i in tr_img_list:
        img = i[:-7]
        if img in list(lbl_dict.keys()):
            tr_data_li.append(i)
    
    te_data_li = []
    for i in te_img_list:
        img = i[:-7]
        if img in list(lbl_dict.keys()):
            te_data_li.append(i)

    return lbl_dict, tr_data_li, te_data_li

def read_data(num_classes):
    tr_root_path = '../data/train'
    tr_img_list = os.listdir(tr_root_path)
    te_root_path = '../data/test'
    te_img_list = os.listdir(te_root_path)
    lbl_dict, tr_data_li, te_data_li = get_labels(tr_img_list, te_img_list, num_classes)

    y_train = np.zeros(len(tr_data_li))
    img_root = os.path.join(tr_root_path, tr_data_li[0])
    x_train = cv2.imread(img_root).reshape(1, -1)
    y_train[0] = lbl_dict[tr_data_li[0][:-7]]
    for idx in range(1, len(tr_data_li)):
        img_name = tr_data_li[idx]
        img_root = os.path.join(tr_root_path, img_name)
        c_np_img = cv2.imread(img_root).reshape(1, -1)
        x_train = np.vstack((x_train, c_np_img))
        y_train[idx] = lbl_dict[img_name[:-7]]

    y_test = np.zeros(len(te_data_li))
    img_root = os.path.join(te_root_path, te_data_li[0])
    x_test = cv2.imread(img_root).reshape(1, -1)
    y_test[0] = lbl_dict[te_data_li[0][:-7]]
    for idx in range(1, len(te_data_li)):
        img_name = te_data_li[idx]
        img_root = os.path.join(te_root_path, img_name)
        c_np_img = cv2.imread(img_root).reshape(1, -1)
        x_test = np.vstack((x_test, c_np_img))
        y_test[idx] = lbl_dict[img_name[:-7]]

    return x_train, x_test[:250], y_train, y_test[:250]

def scale(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    x_train = scaler.transform(train)
    x_test = scaler.transform(test)

    return x_train, x_test, scaler

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False)
    args = parser.parse_args()
    return args

def run(args):
    num_classes = [i for i in range(100)]
    x_train, x_test, y_train, y_test = read_data(num_classes)
    x_train, x_test, scaler = scale(x_train, x_test)

    if args.train == "True":
        print('hi')
        model = MLPClassifier(
        hidden_layer_sizes=(100,), activation='relu',
        solver='adam', alpha=0.001, batch_size=32,
        learning_rate_init=0.001, max_iter=500,
        random_state=123)
        print('start training')
        model.fit(x_train, y_train)
        print('finished training')
        print(f"Accuracy: {model.score(x_test, y_test)}")
    else:
        print('Load model')  
        model = joblib.load("ckpt/model.pkl")
        print(f"Accuracy: {model.score(x_test, y_test)}")

if __name__=="__main__":
    args = parse()
    run(args)