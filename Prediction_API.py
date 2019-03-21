import flask
import numpy as np
import tensorflow as tf
from keras.models import load_model

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
def init():
    global model,graph
    # load the pre-trained Keras model
    model = load_model('trained_model.h5')
    graph = tf.get_default_graph()

# Getting Parameters
def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('age'))
    parameters.append(flask.request.args.get('sex'))
    parameters.append(flask.request.args.get('cp'))
    parameters.append(flask.request.args.get('trestbps'))
    parameters.append(flask.request.args.get('chol'))
    parameters.append(flask.request.args.get('fbs'))
    parameters.append(flask.request.args.get('restecg'))
    parameters.append(flask.request.args.get('thalach'))
    parameters.append(flask.request.args.get('exang'))
    parameters.append(flask.request.args.get('oldpeak'))
    parameters.append(flask.request.args.get('slope'))
    parameters.append(flask.request.args.get('ca'))
    parameters.append(flask.request.args.get('thal'))
    return parameters

# Cross origin support
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

# API for prediction
@app.route("/predict", methods=["GET"])
def predict():
    nameOfTheCharacter = flask.request.args.get('name')
    parameters = getParameters()
    inputFeature = np.asarray(parameters).reshape(1, 13)
    with graph.as_default():
        raw_prediction = model.predict(inputFeature)[0][0]
    if raw_prediction > 0.5:
        prediction = 'Yes'
    else:
        prediction = 'No'
    return sendResponse({nameOfTheCharacter: prediction})

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(threaded=True)