from pathlib import Path
from Cleaner import Cleaner
from ObligationsExtractor import ObligationExtractor
from AiSummarizer import AiSummarizer
import json

def sample_mode():
    for pdf_file in Path('pdf').rglob('*.pdf'):
        Cleaner.pdf_to_txt(pdf_file)
    Cleaner.clean_folder('text')
    extractor = ObligationExtractor(txt_dir='text')
    extractor.save_results(extractor.extract())
    summarizer = AiSummarizer()
    json_files = list(Path('text/json').glob('*.json'))
    for json_file in json_files:
        print(f"Processing {json_file}")
        summarizer.json2txt(json_file)


if __name__ == "__main__":
    sample_mode()