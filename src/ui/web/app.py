from flask import Flask, request, make_response
import numpy as np
import cv2
app = Flask(__name__)
import os

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from classify import classify

@app.route('/')
def hello_world():
    return app.send_static_file('index.html')

@app.route('/classify', methods=['POST'])
def main():
    try:
        model = request.args.get('model', default = 'random_forest', type = str)
        classes = request.args.get('class', default = 'general', type = str)

        img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        _, buffer = cv2.imencode('.png', classify(img, classes,  model))
        return (make_response(buffer.tobytes()), 200, {'Content-Type': 'image/png'})
    except Exception as e:
        return str(e), 404

@app.route('/classifiers', methods=['GET'])
def classifiers():
    def get_models(p):
        return [m.split('.')[0] for m in os.listdir('../../../models/%s' % p)]

    return {
        'general': get_models('general'),
        'specific': get_models('specific')
    }

app.run(debug=True)