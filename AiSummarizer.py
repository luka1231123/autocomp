import argparse
import subprocess
import sys
import json
from pathlib import Path
import replicate

class AiSummarizer:
    def summarize(self, text: str) -> str:
        system_prompt = (
            "You are a highly precise and critical AI assistant. "
            "Your role is to carefully analyze legal and contractual text to determine whether each identified item is a valid and accurate obligation, deadline, payment, or submission requirement, "
            "or if it is mistaken or irrelevant. "
            "Only include information that is correct, relevant, and legally significant."
        )
        prompt = (
            "You are tasked with extracting only valid, clear, and legally significant obligations, deadlines, payments, or submission requirements from the provided text. "
            "Carefully evaluate each potential item to ensure it is accurate, relevant, and not mistaken or out of context. "
            "If no valid item is found, return nothing.\n\n"
            "For each valid item, output strictly in the following format (do not include extra commentary or formatting):\n"
            "event: [a short, precise title describing the obligation or deadline]\n"
            "type: [one of: obligation | deadline | payment | submission]\n"
            "confidence: [integer from 0 to 100 representing how certain you are that this is a valid, well-formed item don't be shy to write low scores]\n"
            "summary: [a detailed but concise summary capturing all legally important details, conditions, and responsibilities]\n"
            "date: [exact date in DD-MM-YYYY format, or leave blank if no specific date is given]\n\n"
            "Process the following text:\n\n" + text
        )

        output = ""

        for event in replicate.stream(
            "openai/gpt-4o-mini",
            input = {
                "prompt": prompt,
                "system_prompt": system_prompt,
            },
        ):
            chunk = event.data
            print(chunk, end="")
            output += chunk

        return output

    def json2txt(self, json_file: str):
        json_path = Path(json_file)
        with open(json_path, "r") as f:
            data = json.load(f)
        textInputs = []
        for item in data:
            sentence = item.get("sentence")
            date = item.get("dates")
            if sentence is not None and date is not None:
                textInput = sentence + " " + " ".join(date)
                textInputs.append(textInput)

        events_dir = Path("events")
        events_dir.mkdir(exist_ok=True)
        out_file = events_dir / ("aiEvent" + json_path.stem + ".txt")
        # delete any previous aiEvent output for this stem
        if out_file.exists():
            out_file.unlink()

        for textInput in textInputs:
            summary = self.summarize(textInput)
            print(f"Summary for {summary}")
            with open(out_file, "a") as out_f:
                out_f.write("\n\n" + summary + "\n\n")