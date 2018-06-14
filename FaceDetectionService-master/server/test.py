import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
smile_cascade = cv.CascadeClassifier('cascades/haarcascade_smile.xml')
eye_cascade = cv.CascadeClassifier('cascades/haarcascade_eye.xml')


def validateFace(face):
    eyes = face['eyes']
    mouth = face['mouth']

    if len(eyes) < 1 or len(eyes) > 2:
        return False

    for m in mouth:
        if m['y'] < eyes[0]['y'] + eyes[0]['height']:
            print('f')
            mouth.remove(m)

    if len(mouth) > 1:
        return False

    return True


def detectFaces(origImg):
    origImg = cv.imread(origImg)

    # cria uma imagem em escala de cinza
    grayImg = cv.cvtColor(origImg, cv.COLOR_BGR2GRAY)
    grayImg = cv.equalizeHist(grayImg)

    # usa a imagem em escala de cinza para encontrar os rostos e coloca num array de tuplas (x, y, width, height)
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 5)

    result = {
        'img': origImg,
        'faces': []
    }

    for (x, y, w, h) in faces:

        # desenha um retângulo azul na imagem original
        cv.rectangle(origImg, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # *roi é um recorte da imagem
        # usa a região da face encontrada e faz um recorte na imagem original e na em escala de cinza
        roi_gray = grayImg[y:y + h, x:x + w]
        roi_color = origImg[y:y + h, x:x + w]

        face = {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'eyes': [],
            'mouth': [],
        }

        # usa o recorte em escala de cinza para buscar sorrisos
        smiles = smile_cascade.detectMultiScale(roi_gray)
        for (sx, sy, sw, sh) in smiles:
            # coloca um retângulo verde na região do sorriso, no recorte colorido da imagem

            face['mouth'].append({
                'x': sx,
                'y': sy,
                'width': sw,
                'height': sh,
            })

            cv.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

        # usa o recorte em escala de cinza para buscar olhos
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # coloca um retângulo amarelo na região do olho, no recorte colorido da imagem
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            face['eyes'].append({
                'x': ex,
                'y': ey,
                'width': ew,
                'height': eh,
            })

        if validateFace(face):
            result['faces'].append(face)

    print(len(result['faces']))

    return result


img = detectFaces('faces.jpg')['img']



