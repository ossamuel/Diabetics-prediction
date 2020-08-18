from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        age = request.form["age"]
        bmi = request.form["bmi"]
        bp = request.form["bp"]
        glu = request.form["glu"]
        return render_template('index.html', success=True, age=age, bmi=bmi, bp=bp, glu=glu)
    else:
        return render_template('index.html', success=False)


if __name__ == "__main__":
    app.run()