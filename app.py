from flask import Flask, render_template, request, jsonify
import os

from processor import ingest_document, agent_dispatcher

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():

    file = request.files["file"]

    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])

    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    file.save(path)

    message = ingest_document(path)

    return jsonify({
        "status": "Success",
        "message": message
    })


@app.route("/ask", methods=["POST"])
def ask():

    query = request.json.get("query", "")

    ans, tool, citations, model_name = agent_dispatcher(query)

    return jsonify({
        "answer": ans,
        "tool": tool,
        "citations": citations,
        "model": model_name
    })


if __name__ == "__main__":
    app.run(debug=True)
