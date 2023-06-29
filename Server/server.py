from flask import Flask, request, json, make_response
from flask_cors import cross_origin
from qabot import QABot

prev_context = ""


def launch(qabot: QABot):
    app = Flask(import_name=__name__)

    @app.post("/")
    @cross_origin()
    def post_():
        global prev_context

        if not request.is_json:
            return "Your request needs to be json.", 500
        if type(request.json)==str:
            req_data = json.loads(request.json)
        else:
            req_data = request.json

        try:
            # Extract Request Data
            context = req_data["context"].replace("\n", " ")
            question = req_data["question"]
            topk = int(req_data["topk"])

            # Execute with Bot
            updated_context = False
            if not context == prev_context:
                updated_context = True
                qabot.set_context(context)
                prev_context = qabot.extract_context(0, -1)
            answers = qabot.ask_question(question, topk)

            # Make Responase Data
            res_data = {}
            res_data["message"] = "Updated context" if updated_context else "Using context buffer"
            res_data["context"] = prev_context
            res_data["answers"] = []
            for answer in answers:
                content = qabot.extract_context(answer.start, answer.end)
                start_in_text = \
                    0 if answer.start == 0 \
                    else len(qabot.extract_context(0, answer.start-1))
                end_in_text = \
                    len(qabot.extract_context(0, answer.end))
                res_data["answers"].append({
                    "content": content,
                    "start": start_in_text,
                    "end": end_in_text,
                    "score": str(answer.score)
                })

            # Return Response
            res = make_response(json.dumps(res_data))
            res.status_code = 200
            return res
        except Exception as e:
            return str(e), 500

    app.run("0.0.0.0", port=3838, debug=False)
