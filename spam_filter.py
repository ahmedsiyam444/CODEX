#!/usr/bin/env python3
"""Train and use a local email spam filter.

This tool implements a multinomial Naive Bayes classifier with Laplace smoothing.
It is dependency-free and can incrementally learn from new labeled emails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class SpamFilterModel:
    """Multinomial Naive Bayes spam classifier."""

    spam_docs: int = 0
    ham_docs: int = 0
    spam_word_counts: Counter[str] = field(default_factory=Counter)
    ham_word_counts: Counter[str] = field(default_factory=Counter)

    @property
    def total_docs(self) -> int:
        return self.spam_docs + self.ham_docs

    @property
    def vocabulary(self) -> set[str]:
        return set(self.spam_word_counts) | set(self.ham_word_counts)

    @property
    def spam_total_words(self) -> int:
        return sum(self.spam_word_counts.values())

    @property
    def ham_total_words(self) -> int:
        return sum(self.ham_word_counts.values())

    def update(self, label: str, text: str) -> None:
        label = normalize_label(label)
        tokens = tokenize(text)
        if label == "spam":
            self.spam_docs += 1
            self.spam_word_counts.update(tokens)
        else:
            self.ham_docs += 1
            self.ham_word_counts.update(tokens)

    def predict_with_score(self, text: str) -> tuple[str, float]:
        if self.total_docs == 0:
            raise ValueError("Model is empty. Train it first.")

        tokens = tokenize(text)
        # Use weak priors if only one class exists so model stays usable.
        spam_prior_count = self.spam_docs if self.spam_docs > 0 else 1
        ham_prior_count = self.ham_docs if self.ham_docs > 0 else 1
        total_prior = spam_prior_count + ham_prior_count

        log_spam = math.log(spam_prior_count / total_prior)
        log_ham = math.log(ham_prior_count / total_prior)

        vocab_size = max(1, len(self.vocabulary))
        spam_den = self.spam_total_words + vocab_size
        ham_den = self.ham_total_words + vocab_size

        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            spam_num = self.spam_word_counts.get(token, 0) + 1
            ham_num = self.ham_word_counts.get(token, 0) + 1
            log_spam += count * math.log(spam_num / spam_den)
            log_ham += count * math.log(ham_num / ham_den)

        # Convert log odds to probability in stable manner.
        max_log = max(log_spam, log_ham)
        spam_exp = math.exp(log_spam - max_log)
        ham_exp = math.exp(log_ham - max_log)
        spam_prob = spam_exp / (spam_exp + ham_exp)

        label = "spam" if spam_prob >= 0.5 else "ham"
        confidence = spam_prob if label == "spam" else (1 - spam_prob)
        return label, confidence

    def to_dict(self) -> dict:
        return {
            "spam_docs": self.spam_docs,
            "ham_docs": self.ham_docs,
            "spam_word_counts": dict(self.spam_word_counts),
            "ham_word_counts": dict(self.ham_word_counts),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SpamFilterModel":
        return cls(
            spam_docs=payload.get("spam_docs", 0),
            ham_docs=payload.get("ham_docs", 0),
            spam_word_counts=Counter(payload.get("spam_word_counts", {})),
            ham_word_counts=Counter(payload.get("ham_word_counts", {})),
        )


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def normalize_label(label: str) -> str:
    value = label.strip().lower()
    if value in {"spam", "1", "true", "yes"}:
        return "spam"
    if value in {"ham", "not_spam", "0", "false", "no"}:
        return "ham"
    raise ValueError(f"Unsupported label '{label}'. Use spam/ham.")


def read_training_rows(path: Path) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"label", "text"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Training CSV must include columns: label,text")

        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or not label:
                continue
            yield label, text


def save_model(model: SpamFilterModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_model(path: Path) -> SpamFilterModel:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return SpamFilterModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


def train_command(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    model = SpamFilterModel()
    trained = 0
    for label, text in read_training_rows(Path(args.data)):
        model.update(label, text)
        trained += 1

    if trained == 0:
        raise ValueError("No valid training rows found.")

    save_model(model, model_path)
    print(f"Trained model with {trained} messages -> {model_path}")
    return 0


def update_command(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    trained = 0
    for label, text in read_training_rows(Path(args.data)):
        model.update(label, text)
        trained += 1

    if trained == 0:
        raise ValueError("No valid training rows found.")

    save_model(model, Path(args.model))
    print(f"Updated model with {trained} additional messages")
    return 0


def predict_command(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    label, confidence = model.predict_with_score(args.text)
    print(json.dumps({"label": label, "confidence": round(confidence, 4)}))
    return 0


def classify_file_command(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    in_path = Path(args.data)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8", newline="") as inf, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as outf:
        reader = csv.DictReader(inf)
        if "text" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must include a text column")

        fieldnames = list(reader.fieldnames or []) + ["prediction", "confidence"]
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            label, confidence = model.predict_with_score(row.get("text", ""))
            row["prediction"] = label
            row["confidence"] = f"{confidence:.4f}"
            writer.writerow(row)

    print(f"Wrote predictions to {out_path}")
    return 0


HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Local Spam Filter</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; line-height: 1.4; }
    textarea, input[type=text] { width: 100%; padding: 8px; }
    textarea { min-height: 80px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    button { padding: 8px 14px; cursor: pointer; }
    .hint { color: #666; font-size: 14px; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Local Email Spam Filter</h1>
  <p class="hint">Model file: <code>{model_path}</code></p>

  <div class="card">
    <h2>Train / Update Model</h2>
    <p class="hint">Paste CSV rows with header <code>label,text</code>. Existing model will be updated if present.</p>
    <form method="POST" action="/train">
      <textarea name="csv_data" placeholder="label,text&#10;spam,Win cash now&#10;ham,Team meeting at 2pm"></textarea>
      <button type="submit">Train/Update</button>
    </form>
  </div>

  <div class="card">
    <h2>Predict One Email</h2>
    <form method="POST" action="/predict">
      <textarea name="text" placeholder="Paste email text here..."></textarea>
      <button type="submit">Predict</button>
    </form>
  </div>

  <p class="hint">API endpoints: <code>POST /api/train</code>, <code>POST /api/predict</code>.</p>
</body>
</html>
"""


def _read_form_data(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length).decode("utf-8")
    return {k: v[0] for k, v in parse_qs(raw).items()}


def _iter_csv_string(csv_data: str) -> Iterable[tuple[str, str]]:
    rows = csv.DictReader(csv_data.splitlines())
    if {"label", "text"} - set(rows.fieldnames or []):
        raise ValueError("CSV must include header: label,text")
    for row in rows:
        label = (row.get("label") or "").strip()
        text = (row.get("text") or "").strip()
        if label and text:
            yield label, text


def serve_command(args: argparse.Namespace) -> int:
    model_path = Path(args.model)

    class Handler(BaseHTTPRequestHandler):
        def _send_html(self, code: int, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, code: int, data: dict) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _load_or_init_model(self) -> SpamFilterModel:
            if model_path.exists():
                return load_model(model_path)
            return SpamFilterModel()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(HTTPStatus.OK, HTML_PAGE.format(model_path=model_path))
                return
            if parsed.path == "/health":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                form = _read_form_data(self)
                if parsed.path in {"/train", "/api/train"}:
                    model = self._load_or_init_model()
                    trained = 0
                    for label, text in _iter_csv_string(form.get("csv_data", "")):
                        model.update(label, text)
                        trained += 1
                    if trained == 0:
                        raise ValueError("No valid rows to train.")
                    save_model(model, model_path)
                    response = {
                        "status": "ok",
                        "trained_rows": trained,
                        "model": str(model_path),
                    }
                    if parsed.path == "/train":
                        self._send_html(
                            HTTPStatus.OK,
                            f"<p>Trained/updated with {trained} rows.</p><p><a href='/'>Back</a></p>",
                        )
                    else:
                        self._send_json(HTTPStatus.OK, response)
                    return

                if parsed.path in {"/predict", "/api/predict"}:
                    model = self._load_or_init_model()
                    text = (form.get("text") or "").strip()
                    if not text:
                        raise ValueError("Missing text")
                    label, confidence = model.predict_with_score(text)
                    response = {"label": label, "confidence": round(confidence, 4)}
                    if parsed.path == "/predict":
                        self._send_html(
                            HTTPStatus.OK,
                            f"<p>Prediction: <b>{label}</b> ({confidence:.4f})</p><p><a href='/'>Back</a></p>",
                        )
                    else:
                        self._send_json(HTTPStatus.OK, response)
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            except Exception as exc:  # broad to return API-safe errors
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Server running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Email spam filter trainer and classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a new model from labeled CSV")
    train.add_argument("--data", required=True, help="CSV with columns: label,text")
    train.add_argument("--model", default="model/spam_model.json", help="Output model JSON path")
    train.set_defaults(func=train_command)

    update = sub.add_parser("update", help="Incrementally train an existing model")
    update.add_argument("--data", required=True, help="CSV with columns: label,text")
    update.add_argument("--model", default="model/spam_model.json", help="Existing model JSON path")
    update.set_defaults(func=update_command)

    predict = sub.add_parser("predict", help="Predict one email text")
    predict.add_argument("--model", default="model/spam_model.json", help="Model JSON path")
    predict.add_argument("--text", required=True, help="Email text to classify")
    predict.set_defaults(func=predict_command)

    batch = sub.add_parser("classify-file", help="Classify all rows in a CSV")
    batch.add_argument("--model", default="model/spam_model.json", help="Model JSON path")
    batch.add_argument("--data", required=True, help="CSV with at least a text column")
    batch.add_argument("--output", default="predictions.csv", help="Output CSV path")
    batch.set_defaults(func=classify_file_command)

    serve = sub.add_parser("serve", help="Run local web UI/API for training and prediction")
    serve.add_argument("--model", default="model/spam_model.json", help="Model JSON path")
    serve.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve.set_defaults(func=serve_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
