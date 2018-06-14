# from flask import Flask
import cv2 as cv

# app = Flask(__name__)

# @app.route("/")
# def hello():
#     return "Face Detection Service"

face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
smile_cascade = cv.CascadeClassifier('cascades/haarcascade_smile.xml')

# carrega uma imagem
origImg = cv.imread('images/serio.jpg')

# cria uma imagem em escala de cinza
grayImg = cv.cvtColor(origImg, cv.COLOR_BGR2GRAY)

# usa a imagem em escala de cinza para encontrar os rostos e coloca num array
faces = face_cascade.detectMultiScale(grayImg, 1.3, 5)

for (x,y,w,h) in faces:
    # desenha um retângulo azul na imagem original
    cv.rectangle(origImg, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # *roi é um recorte da imagem
    # usa a região da face encontrada e faz um recorte na imagem original e na em escala de cinza
    roi_gray = grayImg[y:y + h, x:x + w]
    roi_color = origImg[y:y + h, x:x + w]

    # usa o recorte em escala de cinza para buscar sorrisos
    smiles = smile_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in smiles:
        # coloca um retângulo verde na região do sorriso, no recorte colorido da imagem
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


# exibe imagem
cv.imshow('img', origImg)
cv.waitKey(0)
cv.destroyAllWindows()

