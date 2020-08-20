from flask import Flask, render_template, request
import numpy as np
import pickle
# import sklearn

app = Flask(__name__)

model = pickle.load(open('diabetic_clf_model.plk', 'rb'))
@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        age = request.form["age"]
        bmi = request.form["bmi"]
        bp = request.form["bp"]
        glu = request.form["glu"]

        int_features = [int(float(x)) for x in request.form.values()]
        final = [np.array(int_features)]
        prediction = model.predict(final)
        
        return render_template('index.html', success=True, age=age, bmi=bmi, bp=bp, glu=glu, pred=prediction)
    else:
        return render_template('index.html', success=False)


if __name__ == "__main__":
    app.run(debug=True)
