import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

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
        y[label] = np.int32(1)
        out.append(y.copy())
    return out

def ajuste(data,label):
    data = np.array([feature / 255.0 for feature in data])
    label = np.array(struct_labels(label,4))
    return data,label

def reading_test(dir = './test'):
    import os
    arquivos = [os.path.join(dir, nome) for nome in os.listdir(dir)]
    demiliter = ('/' if arquivos[0][len(dir)]=='/' else '\\')

    arquivos = [caminho.split(demiliter)[-1:][0] for caminho in arquivos ]

    test_data = []
    for fn in arquivos:
        dir_img = dir + '/' + fn
        test_data.append(cv2.imread(dir_img))
    return test_data.copy(),arquivos

def write_predition(arq,predictions,path):
    len_labels = len(predictions)
    with open(path, mode='w', encoding='utf-8') as csv_file:
        i=0
        while(len_labels>i):
            csv_file.writelines(str(arq[i]) + ',' + str(predictions[i]) + '\n')
            i+=1
def write_predition_img(arq,imgs,path):
    len_labels = len(imgs)
    i=0
    while(len_labels>i):
        cv2.imwrite(path + arq[i], imgs[i])
        i+=1


def rotacao_img(img,label):
    if(label==1):
        return img
    if(label==2):
        shape = img.shape
        rotacao = cv2.getRotationMatrix2D((32,32), -90, 1)
        rotacionado = cv2.warpAffine(img, rotacao, (shape[0], shape[1]))
        return rotacionado
    if(label==3):
        shape = img.shape
        rotacao = cv2.getRotationMatrix2D((32,32), 90, 1)
        rotacionado = cv2.warpAffine(img, rotacao, (shape[0], shape[1]))
        return rotacionado
    if(label==4):
        shape = img.shape
        rotacao = cv2.getRotationMatrix2D((32,32), 180, 1)
        rotacionado = cv2.warpAffine(img, rotacao, (shape[0], shape[1]))
        return rotacionado
def array_label(num):
    num_id = 0
    i = 0
    while(i<len(num)):
        if(num[i]>num[num_id]):
            num_id = i
        i+=1
    return num_id
def rotation_predition(test_labels,predictions):
    correct_img = []
    correct_label = []
    for i in range(len(test_labels)):
        label_num = array_label(predictions[i])
        img = rotacao_img(test_labels[i],label_num)
        correct_img.append(img)
        correct_label.append(label_num)
    return correct_img,correct_label


# print(features[0],labels[0])
# train_data, train_label = load_cfar10_batch('cifar-10-batches-py/','data_batch_', '1')
train_data, train_label = load_rot('rotfaces', '')
test_data,arq = reading_test('test')


# features, labels = load_rot('', 'rotfaces')
train_data, train_label = ajuste(train_data, train_label)
test_data = np.array([feature/255.0 for feature in test_data])


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



history = model.fit(train_data,
                    train_label,
                    epochs=1,
                    batch_size=1,
                    validation_split=0.2,
                    verbose=1)

predictions = model.predict(test_data)
imgs,labels = rotation_predition(test_data,predictions)
write_predition_img(arq,imgs,'predition/')
write_predition(arq,labels,'truth.csv')