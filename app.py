from flask import Flask, request, render_template
import requests
from transformers import pipeline

pipe = pipeline(
    "token-classification", model="antoineedy/stanford-deidentifier-base-finetuned-ner"
)

app = Flask(__name__)

HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/antoineedy/stanford-deidentifier-base-finetuned-ner"

HUGGING_FACE_API_KEY = "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"


def postprocess_results(result, input):
    if result == "Error: Could not process the request.":
        return "Error"
    to_return = ""
    for entity in result:
        label = entity["entity"].split("_")[1]
        # word = entity["word"]
        start, end = entity["start"], entity["end"]
        word = input[start:end]
        if len(word) == 1:
            score = ""
            space = ""
        else:
            score = str(round(100 * entity["score"], 2)) + "%"
            space = " "
        to_return += (
            f'<span class="highlight label-{label}">{word}<span class="score">{score}</span></span>'
            + space
        )
    return to_return


def mypipeline(user_input):
    out = pipe(user_input)
    to_remove = []
    for i, entity in enumerate(out):
        if entity["word"][:2] == "##":
            out[i - 1]["end"] = entity["end"]
            out[i - 1]["word"] += entity["word"][2:]
            to_remove.append(i)
    for i in sorted(to_remove, reverse=True):
        out.pop(i)
    return out


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    to_return = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        # headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
        # response = requests.post(
        #    HUGGING_FACE_API_URL, headers=headers, json={"inputs": user_input}
        # )
        result = mypipeline(user_input)
        to_return = postprocess_results(result, user_input)

    return render_template("index.html", result=to_return)


if __name__ == "__main__":
    app.run(debug=True)
