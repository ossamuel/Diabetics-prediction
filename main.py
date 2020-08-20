from flask import Flask, render_template, request
import numpy as np
import pickle
# import sklearn
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        glu = request.form["glu"]
        bp = request.form["bp"]
        bmi = request.form["bmi"]
        age = request.form["age"]
        int_features = [int(float(x)) for x in request.form.values()]
        final = [np.array(int_features)]
        print(final)
        prediction = model.predict_proba(final)
        
        return render_template('index.html', success=True, predict_text = 'your probability of being diabetic is  {} '.format(prediction[0][1]))
    else:
        return render_template('index.html', success=False)


if __name__ == "__main__":
    app.run(debug=True)
