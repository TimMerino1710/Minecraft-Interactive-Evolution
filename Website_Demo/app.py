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
        # print(jsonify(dat))
        return render_template('index.html',house_json_dat=json.dumps(dat))

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == "POST":    
        #import the z set
        zj = json.loads(request.form['z_set'])
        z_set = []
        for k in zj.keys():
            z_set.append(np.array([float(i) for i in list(zj[k])]))

        z_set = np.array(z_set);
        print(z_set.shape)

        #evolve using the z vectors
        if(len(z_set) == 1):
            z, arr = ie.nextEvo(z_set)
        else:
            z, arr = ie.multiNextEvo(z_set)

        dat = {}
        dat['house_arr'] = npy2txt(arr)
        dat['z_set'] = npy2txt(z)

        # print(dat['house_arr'])
        # print(dat['z_set'])
        print(jsonify(dat))
        return render_template('index.html',house_json_dat=json.dumps(dat))

# this goes last
app.run(debug=True)\

