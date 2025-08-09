import re
import pathlib
from pathlib import Path
from PyPDF2 import PdfReader

class Cleaner:
    HEADER_RE  = re.compile(r'^(Revised\s+\d{1,2}/\d{1,2}/\d{2,4}|Page\s+\d+\s+of\s+\d+).*$', re.I)
    BLANK_RE   = re.compile(r'\n{3,}')
    HYPHEN_RE  = re.compile(r'(\w+)-\n(\w+)')
    BULLET_RE  = re.compile(r'^[\s•☐▪–-]+', re.M)
    DOTS_RE    = re.compile(r'\.{2,}')
    HYPHEN_CHAR_RE = re.compile(r'[-–—]')
    LONG_LINE_RE = re.compile(r'^[-–—]{5,}$', re.M)

    @staticmethod
    def pdf_to_txt(pdf_path):
        reader = PdfReader(pdf_path)
        output_dir = Path('text')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / (Path(pdf_path).stem + '.txt')
        if output_path.exists():
            output_path.unlink()
        with open(output_path, 'w', encoding='utf-8') as txt_file:
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    txt_file.write(text + '\n')

    @classmethod
    def clean_text(cls, text: str) -> str:
        lines = [ln for ln in text.splitlines() if not cls.HEADER_RE.match(ln)]
        lines = [ln for ln in lines if not cls.LONG_LINE_RE.match(ln)]
        txt = "\n".join(lines)
        while cls.HYPHEN_RE.search(txt):
            txt = cls.HYPHEN_RE.sub(r'\1\2', txt)
        txt = cls.DOTS_RE.sub('', txt)
        txt = cls.HYPHEN_CHAR_RE.sub('', txt)
        txt = cls.BULLET_RE.sub('- ', txt)
        txt = cls.BLANK_RE.sub('\n\n', txt)
        txt = re.sub(r'[ \t]{2,}', ' ', txt)
        txt = re.sub(r'[ \t]+', ' ', txt)
        txt = re.sub(r'\n[ \t]+', '\n', txt)
        return txt.strip()

    @classmethod
    def clean_folder(cls, path='.', glob='*.txt'):
        p = pathlib.Path(path)
        for f in p.glob(glob):
            cleaned = cls.clean_text(f.read_text(encoding='utf-8', errors='ignore'))
            f.write_text(cleaned, encoding='utf-8')