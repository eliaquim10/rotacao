import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np


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
def encoding_pickle(file,context):
    with open(file, 'rb') as fo:
        # dict = pickle.load(fo, encoding='bytes')
        pickle.dump(fo,context, encoding='bytes')
'''
arquivos = read()
train_data,train_label = reading_train('train',arquivos)
train_data,train_label = np.array(train_data),np.array(train_label)
pickle.dump( [train_data, train_label], open('rotfaces', "wb"))
print("create data")
'''

img = cv2.imread('test/90-890_1981-06-07_2009.jpg')
shape = img.shape

rotacao = cv2.getRotationMatrix2D((32,32), 270, 1)
rotacionado = cv2.warpAffine(img, rotacao, (shape[0], shape[1]))
print(type(rotacionado))
cv2.imshow("45 graus redimensionado", rotacionado)

cv2.waitKey(0)
# cv2.imshow("Espelhado horizontalmente", img)
# inverter = cv2.flip(img, ())
# cv2.imshow("Espelhado horizontalmente", inverter)

# cv2.waitKey(0)
