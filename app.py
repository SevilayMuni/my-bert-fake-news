from flask import Flask, render_template, request
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Directory where your fine-tuned model and tokenizer are saved
model_name = "SevilayG/my-bert-fake-news"

def load_pipeline():
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)
classifier = load_pipeline()

# Create flask app
app = Flask(__name__, template_folder = 'template')

@app.route("/", methods = ["GET"])
def Home():
    return render_template("index.html")

@app.route("/", methods = ["POST"])
def predict():
    text = [str(x) for x in request.form.values()]
    prediction = classifier(text)
    return render_template("index.html", prediction_text = "The news is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug = True)
