# import
from flask import Flask, jsonify, request, render_template
import random
import ie
from npy2txt import npy2txt
import numpy as np
import json

app = Flask(__name__,static_url_path='/static')

# run home page
@app.route('/', methods=['GET', 'POST'])
def home_page():
    return render_template('index.html',house_json_dat='')


#make the first iteration of the set
@app.route('/reset', methods=['GET', 'POST'])
def reset():
    if request.method == "POST":
        dat = {}
        z, arr = ie.startEvo()
        dat['house_arr'] = npy2txt(arr)
        dat['z_set'] = npy2txt(z)

        # print(dat['house_arr'])
        # print(dat['z_set'])
        print(jsonify(dat))
        return render_template('index.html',house_json_dat=json.dumps(dat))



# this goes last
app.run(debug=True)