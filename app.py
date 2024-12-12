from flask import Flask, render_template, request
import numpy
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Directory where your fine-tuned model and tokenizer are saved
model_name = "SevilayG/my-bert-fake-news"
my_model = AutoModelForSequenceClassification.from_pretrained(model_name)
my_tokenizer = AutoTokenizer.from_pretrained(model_name)
# Create a text classification pipeline
clf = pipeline("text-classification", model = my_model, tokenizer = my_tokenizer)

# Create flask app
app = Flask(__name__, template_folder='template')

@app.route("/", methods = ["GET"])
def Home():
    return render_template("index.html")

@app.route("/", methods = ["POST"])
def predict():
    text = [str(x) for x in request.form.values()]
    prediction = clf(text)
    return render_template("index.html", prediction_text = "The news is {}".format(prediction))

if __name__ == "__main__":
    app.run()
