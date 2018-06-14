import requests
import mimetypes
import json

server_addr = 'http://localhost:5000'
face_detector_url = server_addr + '/face_detector'

while True:
    img_path = input('Informe a imagem: ')

    if img_path == '0':
        exit()

    try:
        # prepare headers for http request
        mime_type = mimetypes.guess_type(img_path)
        headers = {'content-type': mime_type[0]}

        # send http request with image and receive response
        param = {'url': img_path}
        response = requests.get(face_detector_url, params=param, headers=headers)

        # decode response
        response_json = json.loads(response.text)
        print(json.dumps(response_json, indent=2))
    except Exception as e:
        print(e)
