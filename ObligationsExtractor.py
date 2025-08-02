from pathlib import Path
import re, json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest


OBLIGATION_TOKENS = [
    # core modal verbs
    "shall", "must", "should", "will", "has to", "have to", "had to",

    # legal “require” family
    "require", "requires", "required", "requirement", "be required to",
    "is required to", "are required to", "shall be required to",

    # duty & obligation words
    "duty", "duties", "obligation", "obligations", "obligated", "obligate",
    "is obligated to", "are obligated to", "undertake to", "undertakes to",

    # compliance / conformance
    "comply", "complies", "complied", "compliance", "in compliance with",
    "conform", "conforms", "conformance", "in conformance with",
    "pursuant to", "in accordance with", "consistent with",

    # affirmative verbs
    "ensure", "ensures", "ensured", "warrant", "warrants", "guarantee",
    "guarantees", "guaranteed", "maintain", "maintains", "maintained",
    "submit", "submits", "submitted", "provide", "provides", "provided",
    "deliver", "delivers", "delivered", "furnish", "furnishes", "furnished",
    "obtain", "obtains", "obtained", "retain", "retains", "retained",
    "notify", "notifies", "notified", "report", "reports", "reported",
    "pay", "pays", "paid", "remit", "remits", "remitted",
    "perform", "performs", "performed", "carry out", "carries out",
    "carry-out",  # alternative hyphenation

    # time-pressure phrases
    "no later than", "not later than", "within", "on or before",
    "prior to", "immediately", "promptly", "forthwith", "without delay",
    "as soon as practicable", "as soon as reasonably practicable",

    # negative prohibitions (still obligations)
    "shall not", "must not", "may not", "is prohibited from",
    "are prohibited from", "is forbidden to", "are forbidden to",
    "prohibit", "prohibits", "prohibited", "ban", "bans", "banned",

    # conditional / triggering language
    "upon receipt of", "upon request", "upon approval",
    "if requested", "if required", "where applicable",

    # responsibility words
    "responsible for", "responsibility to", "liability to", "liable for",

    # certification / attest
    "certify", "certifies", "certified", "attest", "attests", "attested",

    # inspection / audit
    "inspection", "inspect", "inspects", "inspected",
    "audit", "audits", "audited", "review", "reviews", "reviewed",

    # record-keeping
    "record", "records", "recorded", "record-keeping", "keep records of",
    "retain records", "retain documentation",

    # payment & financial
    "invoice", "invoices", "invoiced", "payable", "payment due",
    "fees due", "fee schedule", "cost recovery",

    # environmental / safety duties
    "monitor", "monitors", "monitored", "measure", "measures", "measured",
    "mitigate", "mitigates", "mitigated", "protect", "protects", "protected",
    "safeguard", "safeguards", "secured", "compensate", "compensates",
    "compensated",

    # catch-all compliance triggers
    "as required by law", "as required herein", "as required under",
    "subject to", "subject to the following", "to the extent necessary",
]

DATE_TOKENS = [
    # numeric only
    r"\b\d{4}-\d{2}-\d{2}\b",               # 2025-08-02
    r"\b\d{2}/\d{2}/\d{4}\b",               # 02/08/2025
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", # 2-8-25, 2.8.2025

    # month name first
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*[ .-]+\d{1,2}[, ]+\d{2,4}\b",                        # Aug 2 2025
    r"\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)"
    r"[ .-]+\d{1,2}(?:st|nd|rd|th)?[, ]+\d{4}\b",               # August 2nd, 2025

    # day first
    r"\b\d{1,2}(?:st|nd|rd|th)?[ .-]+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*[ .-]+\d{2,4}\b",                                   # 2 Aug 25
    r"\b\d{1,2}(?:st|nd|rd|th)?[ .-]+"
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)"
    r"[ ,.-]+\d{4}\b",                                          # 2nd August 2025

    # ISO / week & ordinal variants
    r"\b\d{4}W\d{2}\b",         # 2025W32 (ISO week)
    r"\b\d{4}-\d{3}\b",         # 2025-214  (ordinal date)
    r"\bQ[1-4]\s*\d{4}\b",      # Q3 2025
]

DATE_RE = re.compile("|".join(DATE_TOKENS), flags=re.I)


class ObligationExtractor:
    def __init__(self, txt_dir='text', contamination=0.10):
        self.txt_dir = Path(txt_dir)
        self.contamination = contamination
        self.vec  = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)
        self.iso  = IsolationForest(contamination=contamination, random_state=42)

    def _sentences(self):
        sents, files = [], []
        for p in self.txt_dir.glob('*.txt'):
            for s in re.split(r'[.\n]+', p.read_text(errors='ignore')):
                if len(s.split()) >= 4:
                    sents.append(s.strip()); files.append(p.name)
        return sents, files

    def extract(self):
        sents, files = self._sentences()
        X = self.vec.fit_transform(sents)
        self.iso.fit(X)
        scores = self.iso.decision_function(X)
        thresh = np.quantile(scores, self.contamination)
        results = [
            {'file': f, 'sentence': s, 'dates': DATE_RE.findall(s)}
            for s, f, sc in zip(sents, files, scores)
            if sc <= thresh and any(t in s.lower() for t in OBLIGATION_TOKENS)
        ]
        return results
    def save_results(self, results):
        json_dir = self.txt_dir / 'json'
        json_dir.mkdir(exist_ok=True)
        file_obligations = {}
        for item in results:
            file_obligations.setdefault(item['file'], []).append({
                'sentence': item['sentence'],
                'dates': item['dates'] or []
            }) 
        for fname, obligations in file_obligations.items():
            out_path = json_dir / (fname[:-4] + '.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(obligations, f, indent=2, ensure_ascii=False)
