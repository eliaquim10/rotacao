import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
import sys

def read(dir = 'train.truth.csv'):
    arquivos = {}
    labels = ['upright', 'rotated_left', 'rotated_right', 'upside_down']
    labels_num = lambda label,labels:[l for l in range(len(labels)) if(labels[l]==label)]
    with open(dir) as imgs:
        next(imgs)
        for line in imgs:
            (fn ,label) = line.split(',')
            arquivos[fn] = labels_num(label[:-1],labels)[0]
    return arquivos

def reading_train(dir,arquivos):
    train_data = []
    train_label = []
    #Read Image
    for fn,label in arquivos.items():
        try:
            img = cv2.imread(dir + '/' + fn)
            if(None not in img):
                train_data.append(img)
                train_label.append(label)
        except Exception:
            pass
    return train_data,train_label

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

if (__name__ == '__main__'):
    train,pickle_train = sys.argv[1],sys.argv[2]
    arquivos = read(train)
    train_data,train_label = reading_train('train',arquivos)
    train_data,train_label = np.array(train_data),np.array(train_label)
    pickle.dump( (train_data, train_label), open(pickle_train, "wb"))
    print("create data")

# python util_rotacao.py train.truth.csv rotfaces