from qabot import QABot
from answer import Answer
from typing import List, Tuple
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch


class DistilBERTQABot(QABot):
    def __init__(self, modeldir: str) -> None:
        super().__init__()
        self.model = DistilBertForQuestionAnswering.from_pretrained(modeldir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(modeldir)

        self.set_context("")

    def tokenize(self, *text: List[str], trim=False) -> Tuple:
        inputs = self.tokenizer(*text, return_tensors="pt")
        inputids = inputs["input_ids"][0].numpy()
        tokens = [self.tokenizer.ids_to_tokens[i]for i in inputids]
        if trim:
            inputids = inputids[1:-1]
            tokens = tokens[1:-1]
        return inputs, inputids, tokens

    def set_context(self, context: str):
        self.c_text = context
        self.c_inputs, self.c_inputids, self.c_tokens = self.tokenize(
            context, trim=True)
        self.num_c_token = len(self.c_inputids)

    def extract_context(self, start: int, end: int) -> str:
        if end < 0:
            end = self.num_c_token
        ids = self.c_inputids[start:end+1]
        return self.tokenizer.decode(ids)
    
    def ask_question(self, question: str, topk: int) -> List[Answer]:
        inputs, inputids, tokens = self.tokenize(question, self.c_text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        start_logits = outputs.start_logits[0, -self.num_c_token-1:-1].numpy()
        end_logits = outputs.end_logits[0, -self.num_c_token-1:-1].numpy()

        pred_data = list()
        for s, s_logit in enumerate(start_logits):
            for e, e_logit in enumerate(end_logits[s+1:], s+1):
                logit = s_logit+e_logit
                pred_data.append((logit, s, e))

        topk_pred_data = sorted(
            pred_data, key=lambda data: data[0], reverse=True)[:topk]
        answers = [
            Answer(logit, s, e) for logit, s, e in topk_pred_data
        ]
        return answers
