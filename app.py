from flask import Flask, render_template, request, redirect, url_for
import pickle
from text_utils import preprocessing

app = Flask(__name__)


with open("News_detection_SVM.pkl", 'rb') as f:
    model = pickle.load(f)

with open("TfidfVectorizer.pkl", 'rb') as f:
    vectoriser = pickle.load(f)

def transformation(text):
    cleaned_text = preprocessing(text)

    X_input = vectoriser.transform([cleaned_text])

    return X_input

@app.route("/True")
def true():
    return render_template("true.html")

@app.route("/false")
def false():
    return render_template("false.html")


@app.route('/', methods = ["POST", "GET"])
def form():
    if request.method == "GET":
        return render_template('form.html')
    else:
        news = str(request.form['news'])
        
        prediction = model.predict(transformation(news))[0]

        if prediction == 0:
            return redirect(url_for("false"))
        else:
            return redirect(url_for("true"))




        



