import cv2
import sys
import matplotlib.pyplot as plt
from util_rotacao import reading_test,read
def rotacao_img(img,label):
    if(label==1):
        return img
    if(label==2):
        shape = img.shape
        rotacao = cv2.getRotationMatrix2D((32,32), 90, 1)
        rotacionado = cv2.warpAffine(img, rotacao, (shape[0], shape[1]))
        return rotacionado
    if(label==3):
        shape = img.shape
        rotacao = cv2.getRotationMatrix2D((32,32), -90, 1)
        rotacionado = cv2.warpAffine(img, rotacao, (shape[0], shape[1]))
        return rotacionado
    if(label==4):
        shape = img.shape
        rotacao = cv2.getRotationMatrix2D((32,32), 180, 0)
        rotacionado = cv2.warpAffine(img, rotacao, (shape[0], shape[1]))
        return rotacionado
def rotation_predition(test_data,arq,predictions):
    correct_img = []
    value = list(predictions.values())
    for i in range(len(test_data)):
        img = rotacao_img(test_data[i],value[i])
        correct_img.append(img)
    return correct_img

def write_predition_img(arq,imgs,path):
    len_labels = len(imgs)
    i=0
    while(len_labels>i):
        cv2.imwrite(path+ '/'+ arq[i], imgs[i])
        i+=1
if (__name__ == '__main__'):
    file_imgs,pred_name,save_predition = sys.argv[1],sys.argv[2],sys.argv[3]
    test_data,arq = reading_test(file_imgs)
    predictions = read(pred_name)
    imgs = rotation_predition(test_data,arq,predictions)
    write_predition_img(arq,imgs,save_predition)


# python girar.py test truth.csv predition