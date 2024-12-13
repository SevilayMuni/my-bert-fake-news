from flask import Flask, render_template, request
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Directory where your fine-tuned model and tokenizer are saved
model_name = "SevilayG/my-bert-fake-news"

# Label dictionary
label_mapping = {
    "LABEL_0": "Fake News",
    "LABEL_1": "True News"
}

def load_pipeline():
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

classifier = load_pipeline()

# Create Flask app
app = Flask(__name__, template_folder='template')

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict():
    try:
        # Get input text
        text = request.form.get("news_text")
        if not text.strip():
            return render_template("index.html", prediction_text="Please enter valid news text.")

        # Perform classification
        prediction = classifier([text])[0]
        label = label_mapping.get(prediction['label'], "Unknown")
        confidence = prediction['score'] * 100

        # Format result
        result = f"Prediction: {label}\nConfidence: {confidence:.2f}%"
        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
