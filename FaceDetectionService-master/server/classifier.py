import cv2 as cv
import numpy as np
import base64

debug = False


def put_label(image, position, text, draw_id):
    font_face = cv.FONT_HERSHEY_DUPLEX
    font_scale = .5
    blue = (255, 0, 0)
    white = (255, 255, 255)
    thickness = 1
    line_type = cv.LINE_AA

    (x, y, w, h) = position
    cv.rectangle(image, (x, y), (x + w, y + h), blue, 2)
    if draw_id:
        cv.rectangle(image, (x - 1, y - 18), (x + w + 1, y), blue, cv.FILLED)
        cv.putText(image, text, (x + 2, y - 4), font_face, font_scale, white, thickness, line_type)


def create_image_json(img):
    _, jpg_img = cv.imencode('.jpg', img)

    return {
        'base64_image': "data:image/jpeg;base64,{}".format(base64.b64encode(jpg_img).decode("utf-8")),
        'resolution': "{}x{}".format(img.shape[1], img.shape[0])
    }


def image_to_cv(image_data):
    # convert string of image data to uint8
    np_array = np.fromstring(image_data, np.uint8)

    # decode image
    return cv.imdecode(np_array, cv.IMREAD_COLOR)


def image_base64_to_cv(image_data):
    encoded_data = image_data.split(',')[1]

    # convert string of image data to uint8
    np_array = np.fromstring(base64.b64decode(encoded_data), np.uint8)

    # decode image
    return cv.imdecode(np_array, cv.IMREAD_COLOR)


def detect_faces(origin_image, is_base64=False, show_ids=False):
    face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    
    if is_base64:
        image = image_base64_to_cv(origin_image)
    else:
        image = image_to_cv(origin_image)

    face_list = []
    response = dict()
    response['faces_detected'] = 0

    # cria uma imagem em escala de cinza
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_img = cv.equalizeHist(gray_img)

    # usa a imagem em escala de cinza para encontrar os rostos e coloca num array
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    i = 0
    for (x, y, w, h) in faces:
        put_label(image, (x, y, w, h), "id: {}".format(i), show_ids)
        face_list.append({
            'id': i,
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h)
        })
        i += 1

    # caso encontre alguma face
    if i > 0:
        response['faces_detected'] = i
        response['face_list'] = face_list
        response['result_image'] = create_image_json(image)

        if debug:
            encoded_img = base64.b64decode(response['result_image']['base64_image'].encode("utf-8"))
            np_array = np.fromstring(encoded_img, np.uint8)
            i = cv.imdecode(np_array, cv.IMREAD_COLOR)
            cv.imshow('img', i)
            cv.waitKey(0)
            cv.destroyAllWindows()

    return response


# debug local - exibe imagem
# if debug:
#     img = open('images/3x4.jpg', 'rb').read()
#     print(detect_faces(img))
