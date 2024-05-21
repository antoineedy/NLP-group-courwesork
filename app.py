from flask import Flask, request, render_template
import requests

app = Flask(__name__)

HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/antoineedy/stanford-deidentifier-base-finetuned-ner"

HUGGING_FACE_API_KEY = "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"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    to_return = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
        response = requests.post(
            HUGGING_FACE_API_URL, headers=headers, json={"inputs": user_input}
        )
        if response.status_code == 200:
            result = response.json()
        else:
            result = "Error: Could not process the request."
        to_return = ""
        for entity in result:
            if entity["entity_group"] == "LABEL_0":
                to_return += f'<span style="color: red;">{entity["word"]}</span> '
            elif entity["entity_group"] == "LABEL_1":
                to_return += f'<span style="color: blue;">{entity["word"]}</span> '
            else:
                to_return += f'<span style="color: green;">{entity["word"]}</span> '

    return render_template("index.html", result=to_return)


if __name__ == "__main__":
    app.run(debug=True)
