import argparse
import subprocess
import sys
import json

class AiSummarizer:
    def __init__(self, model: str = "llama2:7b"):
        self.model = model
    def summarize(self, text: str) -> str:
        prompt = (
            "You are an AI assistant that extracts calendar events from unstructured text. "
            "Identify obligations, deadlines, payments, or submissions and output in the following format, and nothing else:\n"
            "event: [title]\n"
            "type: [type]\n"
            "summary: [long summary]\n"
            "date: [ISO 8601 date]\n\n"
            "Process the following text:\n\n" + text
        )
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Ollama call failed:\n{e.stderr}", file=sys.stderr)
            sys.exit(e.returncode)
    
    def json2txt(self, json_file: str):
        with open(json_file, "r") as f:
            data = json.load(f)
        textInputs = []
        for item in data:
            sentence = item.get("sentence")
            date = item.get("dates")
            if sentence is not None and date is not None:
                textInput= sentence + " " + " ".join(date)
                textInputs.append(textInput)

        for textInput in textInputs:
            summary = self.summarize(textInput)
            print(f"Summary for {summary}\n")
            out_file = "events/aiEvent" + json_file.stem + ".txt"
            with open(out_file, "a") as out_f:
                out_f.write(summary + "\n\n")
"""
    def example_usage(self):
        example_text = (
            "On August 2, 2025, the annual company meeting will be held at the main office. "
            "All employees are expected to attend and participate in the discussions regarding "
            "the upcoming projects and budget allocations."
        )
        summary = self.summarize(example_text)
        print(f"Example Summary:\n{summary}")
"""
