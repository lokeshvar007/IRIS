from flask import Flask, request, render_template
import numpy as np 
import pandas as pd
import pickle

app = Flask(__name__)
sm = pickle.load(open('model1.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    ''' 
    for rendering results on HTML GUI
    '''

    int_feature  = [float(x) for x in request.form.values()]
    final_features = [np.array(int_feature)]
    prediction = sm.predict(final_features)

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text = '' .format(float(prediction)))

if __name__ == "__main__":
    app.run(debug = True)