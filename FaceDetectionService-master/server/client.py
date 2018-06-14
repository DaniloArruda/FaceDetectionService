import requests
import mimetypes
import json

server_addr = 'http://localhost:5000'
face_detector_url = server_addr + '/api/detect/from-file'

while True:
    img_path = input('Informe a imagem: ')

    if img_path == '0':
        exit()

    try:
        # img_path = 'images/3x4.jpg'
        img = open(img_path, 'rb').read()

        # prepare headers for http request
        mime_type = mimetypes.guess_type(img_path)
        headers = {'content-type': mime_type[0]}

        params = {'is_base64': False, 'file': img}

        # send http request with image and receive response
        response = requests.get(face_detector_url, params=params, headers=headers)

        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(e)
