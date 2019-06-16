import cv2
import sys
from util_rotacao import reading_test,read,keys
def rotacao_img(img,label):
    shape = img.shape
    dimension = tuple(shape[:2])
    ponto = (dimension[0]/2,dimension[1]/2)
    scale = 1
    angulo = ([0,-90,90,180][label])
    if(label==0):
        return img
    rotacao = cv2.getRotationMatrix2D(ponto, angulo, scale)
    return cv2.warpAffine(img, rotacao, dimension)
def rotation_predition(test_data,predictions):
    correct_img = []
    for file, data in test_data.items():
        img = rotacao_img(data,predictions[file])
        correct_img.append(img)
    return correct_img

def write_predition_img(arq,imgs,path):
    len_labels = len(imgs)
    i=0
    while(i<len_labels):
        cv2.imwrite(path+ '/'+ arq[i], imgs[i])
        i+=1
if (__name__ == '__main__'):
    file_imgs,pred_name,save_predition = sys.argv[1],sys.argv[2],sys.argv[3]
    test_data = reading_test(file_imgs)
    predictions = read(pred_name)
    imgs = rotation_predition(test_data,predictions)
    write_predition_img(keys(test_data),imgs,save_predition)


# python girar.py test truth.csv predition