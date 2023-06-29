from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser("DistilBERT QABot Server", description="Serves QABot services using DistilBERT")
    parser.add_argument("-m", required=True, help="Path to the DistilBERT QA Model", type=Path)
    args = parser.parse_args()
    modeldir = args.m.resolve()
    
    from distilbertqabot import DistilBERTQABot
    qabot = DistilBERTQABot(str(modeldir))
    from server import launch
    launch(qabot)