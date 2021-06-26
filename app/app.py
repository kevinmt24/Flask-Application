from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
import uuid
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/basepic.svg')
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".svg"
        model = load('model.joblib')
        np_arr = floats_string_to_np_arr(text)
        make_picture('AgesAndHeights.pkl', model, np_arr, path)

        return render_template('index.html', href=path)


def make_picture(training_data_filename, model,new_input_np_arr, output_file):
  data = pd.read_pickle(training_data_filename)
  ages = data['Age']
  data = data[ages > 0]
  data['Height'] = data['Height'].multiply(2.54)
  height = data['Height']
  ages = data['Age']
  #Predictions
  X_new = np.array(list(range(19))).reshape(19,1)
  preds = model.predict(X_new)

  #Previous Figure
  figure = px.scatter(data, x = ages, y = height, title = 'Height vs Age of People', labels = {'x' : 'Ages in Years', 'y' : 'Height in cm'})
  figure.add_trace(go.Scatter(x= X_new.reshape(19), y = preds, mode = 'lines', name = 'Model'))
  
  #New Input Prediction
  new_preds = model.predict(new_input_np_arr)
  figure.add_trace(go.Scatter(x = new_input_np_arr.reshape(len(new_input_np_arr)),y = new_preds, name = "New Outputs",mode = 'markers',marker = dict(color = 'purple',size = 15, line =dict(color = 'purple',width = 2))))
  figure.show()
  figure.write_image(output_file,width = "800",engine = "kaleido")

def floats_string_to_np_arr(floats_str):
    def isfloat(s):
        try:
            float(s)
            return True
        except:
            return False
    floats = np.array([float(x) for x in floats_str.split(',')if isfloat(x)])
    return floats.reshape(len(floats),1) 