import json
import urllib.request
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import classifier

# Initialize the Flask application
app = Flask(__name__)
CORS(app)


# route http posts to this method
@app.route('/api/detect/from-file', methods=['GET'])
def get_json_from_file():
    img = request.args.get('file')
    if img is None:
        return Response(response="Nenhum dado recebido", status=400)
    try:

        response = classifier.detect_faces(img, is_base64=True, show_ids=True)
        response_json = json.dumps(response)
        return Response(response=response_json, status=200, mimetype="application/json")
    except:
        return Response(response="Formato de imagem não suportado", status=415)


# rota para retornar uma imagem usando get
@app.route('/api/detect/from-url', methods=['GET'])
def get_json_from_url():
    try:
        img_url = request.args.get('url')
        img = urllib.request.urlopen(img_url).read()
    except:
        return Response(response="URL inválido", status=400)

    try:
        response = classifier.detect_faces(img, is_base64=False, show_ids=True)
        response_json = json.dumps(response)
        return Response(response=response_json, status=200, mimetype="application/json")
    except:
        return Response(response="Formato de imagem não suportado", status=415)


@app.route('/page')
def render_page():
    return render_template('index.html')


# start flask app
app.run(debug=True)
