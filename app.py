from flask import Flask, request, render_template
import requests
from transformers import pipeline
import time

from tabulate import tabulate


pipe = pipeline(
    "token-classification", model="antoineedy/stanford-deidentifier-base-finetuned-ner"
)

app = Flask(__name__)

HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/antoineedy/stanford-deidentifier-base-finetuned-ner"

HUGGING_FACE_API_KEY = "****"


def create_logs(result, input):

    to_log = ""

    date = time.strftime("%Y-%m-%d %H:%M:%S")
    to_log += f"------- {date} -------\n"
    to_log += f"-> Input: {input}\n"
    to_log += "-> Output:\n"

    words = []
    labels = []
    scores = []
    for entity in result:
        label = entity["entity"].split("_")[1]
        # word = entity["word"]
        start, end = entity["start"], entity["end"]
        word = input[start:end]
        if len(word) == 1:
            score = ""
        else:
            score = str(round(100 * entity["score"], 2)) + "%"
        words.append(word)
        labels.append(label)
        scores.append(score)
    tomap = ["B-O", "B-AC", "B-LF", "I-LF"]
    labels = [tomap[int(label)] for label in labels]
    all = [words, labels, scores]
    table = tabulate(
        tabular_data=all, numalign="center", stralign="center", tablefmt="rounded_grid"
    )
    to_log += str(table)
    to_log += "\n\n"
    return to_log


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
    logs = create_logs(result, input)
    # write a new line in log.txt
    with open("log.txt", "a") as f:
        f.write(logs)
    return to_return


def mypipeline(user_input):
    out = pipe(user_input)
    to_remove = []
    for i, entity in enumerate(out):
        if entity["word"][:2] == "##" and i not in to_remove:
            odxs = [i]
            for j in range(i + 1, len(out)):
                if out[j]["word"][:2] == "##":
                    odxs.append(j)
                else:
                    break
            out[i - 1]["end"] = out[odxs[-1]]["end"]
            word = ""
            for j in odxs[::-1]:
                word += out[j]["word"][2:]
                to_remove.append(j)
            out[i - 1]["word"] += word

    for i in sorted(to_remove, reverse=True):
        out.pop(i)
    return out


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    to_return = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        result = mypipeline(user_input)
        to_return = postprocess_results(result, user_input)

    return render_template("index.html", result=to_return)


if __name__ == "__main__":
    app.run(debug=True)
