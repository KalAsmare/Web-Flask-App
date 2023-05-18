from flask import Flask, request, url_for, redirect, render_template
import pickle
#import xgboost

import numpy as np

app = Flask(__name__, template_folder='./templates', static_folder='./static')

Pkl_Filename = "model4.pkl"
with open(Pkl_Filename, 'rb') as f:
    model = pickle.load(f)
@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    features = [int(x) for x in request.form.values()]

    print(features)
    final = np.array(features).reshape((1, 9))
    print(final)
    pred = model.predict(final)[0]
    print(pred)

    if pred == 1:
        res_val = "** cervical cancer **"
    else:
        res_val = "❤❤no cervical cancer❤❤"

    return render_template('op.html', pred='Patient has {}'.format(res_val))


if __name__ == '__main__':
    app.run(debug=True)
