# Email Spam Filter (Train on Your Own Emails)

This project gives you a **working spam filter** that can learn from your own labeled email list.

## What it does

- Trains a local spam/ham model from a CSV file.
- Predicts if one email is spam.
- Classifies an entire CSV file.
- Supports incremental retraining when you provide more emails.
- Includes a localhost web UI/API so you can use it in browser.

No external Python packages are required.

## 1) Prepare your training data

Create a CSV file like this (`emails_train.csv`):

```csv
label,text
spam,"You won a free iPhone, click now!"
ham,"Can we move our meeting to 2pm tomorrow?"
spam,"Limited-time loan offer with no credit check"
ham,"Here is the invoice for March services"
```

Allowed labels:
- `spam` (or `1`, `true`, `yes`)
- `ham` (or `0`, `false`, `no`, `not_spam`)

## 2) Train a model

```bash
python3 spam_filter.py train --data emails_train.csv --model model/spam_model.json
```

## 3) Predict one message

```bash
python3 spam_filter.py predict --model model/spam_model.json --text "Congratulations! Claim your cash prize now"
```

Output example:

```json
{"label": "spam", "confidence": 0.9342}
```

## 4) Classify a file of incoming emails

Input CSV example (`incoming.csv`):

```csv
id,text
1,"Reminder: your appointment is tomorrow"
2,"Urgent: verify account now to avoid suspension"
```

Run:

```bash
python3 spam_filter.py classify-file --model model/spam_model.json --data incoming.csv --output predictions.csv
```

`predictions.csv` will include:
- original columns
- `prediction`
- `confidence`

## 5) Teach the model with more emails later

```bash
python3 spam_filter.py update --data more_labeled_emails.csv --model model/spam_model.json
```

## 6) Run on localhost (browser URL)

Start local server:

```bash
python3 spam_filter.py serve --host 127.0.0.1 --port 8000
```

Open in browser:

```text
http://127.0.0.1:8000
```

API endpoints:
- `POST /api/train` with form field `csv_data` (`label,text` CSV content)
- `POST /api/predict` with form field `text`

This app is fully local, so your email data stays on your machine.
