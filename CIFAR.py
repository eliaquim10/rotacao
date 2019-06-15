import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from util_rotacao import reading_test
import sys

def load_cfar10_batch(cifar10_dataset_folder_path,data_batch, batch_id):
    with open(cifar10_dataset_folder_path + data_batch + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels
def load_rot(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + str(batch_id), mode='rb') as file:
        (data, labels) = pickle.load(file, encoding='bytes')
    return data, labels

def struct_labels(labels,num_labels):
    out = []
    for label in labels:
        y = np.zeros(num_labels)
        y[label] = 1
        out.append(y.copy())
    return out

def ajuste(data,label):
    data = np.array([feature / 255.0 for feature in data])
    label = np.array(struct_labels(label,4))
    return data,label



def write_predition(arq,predictions,path):
    len_labels = len(predictions)
    def labels_class(i):
        labels = ['upright', 'rotated_left', 'rotated_right', 'upside_down']
        return labels[i]
    with open(path, mode='w', encoding='utf-8') as csv_file:
        i=0
        csv_file.writelines('fn,label\n')
        while(len_labels>i):
            csv_file.writelines(str(arq[i]) + ',' + str(labels_class(predictions[i])) + '\n')
            i+=1

def array_label(num):
    num_id = 0
    i = 0
    while(i<len(num)):
        if(num[i]>num[num_id]):
            num_id = i
        i+=1
    return num_id



# print(features[0],labels[0])
# train_data, train_label = load_cfar10_batch('cifar-10-batches-py/','data_batch_', '1')

def train_test(train_data,train_label,test_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(6,6),strides=(1,1),padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(3,3),
        # tf.keras.layers.Conv2D(128,(5,5),activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax)])

# model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  # loss_weights=[0.01],
                  metrics=['accuracy'])



    model.fit(train_data,
                        train_label,
                        epochs=1,
                        batch_size=1,
                        validation_split=0.2,
                        verbose=1)

    predictions =  model.predict(test_data)
    return [array_label(predictions[i]) for i in range(len(predictions))]
if (__name__ == '__main__'):
    train,test_name,test_predition = sys.argv[1],sys.argv[2],sys.argv[3]
    train_data, train_label = load_rot(train, '')
    test_data,arq = reading_test(test_name)

    train_data, train_label = ajuste(train_data, train_label)
    test_data = np.array([feature/255.0 for feature in test_data])
    labels = train_test(train_data,train_label,test_data)
    print(labels[0])
    write_predition(arq,labels,test_predition)

# python CIFAR.py rotfaces test truth.csv