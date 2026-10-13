# Evaluation

Measures how accurately the pipeline extracts fields, and whether it routes
the right documents to human review, on a labeled dataset that covers plain
text, text-layer PDFs and scanned PDFs.

Results for each run are in [results/README.md](results/README.md).

## Dataset

44 synthetic documents in [data/](data/), built by `build_dataset.py` from a
fixed seed. Running it again writes byte-identical files. All names,
companies, addresses and 555 phone numbers are fictional.

| Type | Count | Missing one required field |
|---|---|---|
| Invoice | 12 | 3: due date, invoice number, currency |
| Resume | 9 | 2: phone, summary |
| Contract | 9 | 2: expiration date, governing law |
| Other (recipe, meeting notes, building notice) | 3 | n/a |
| Poor-scan copies of the 11 scanned documents | 11 | the same as the originals |

Each type is split evenly across three formats: plain text, PDF with a text
layer, and scanned PDF. Scans are rendered at 115 DPI, slightly rotated,
blurred and JPEG-compressed, with no text layer, so only the vision model can
read them.

The 11 scanned documents appear a second time as poor scans: 80 DPI, up to 2°
of skew, heavier blur and JPEG quality 30, about what a low-resolution fax or
a hurried phone photo gives. A person can still read them, but 3s, 6s and 8s
start to look alike. The labels are the same as the originals', so the gap
between the Scanned PDF and Poor scan columns is the effect of scan quality
alone.

A document missing a required field should be sent to human review, and so
should every document of no supported type. Everything else should be stored
without review.

Contracts are service, consulting, license, lease and supply agreements, all
of which carry payment terms. NDAs are left out because the pipeline treats
payment terms as required for every contract (see Known Limitations in the
main README).

## Scoring

`scoring.py` compares each ground-truth field with what the pipeline
extracted. Formatting differences don't count as errors; value differences
always do:

- dates match in any common printed format
- amounts match within one cent
- company names match without suffixes like LLC or Inc.
- phone numbers match on digits
- skills match when the overlap F1 is at least 0.8
- line items match on count and per-line totals
- experience, education and contract parties match as sets of names
- free-text fields (summary, key terms, payment terms, termination clause)
  only need to be present. Judging their wording would need a second model as
  a judge, and that model's errors would end up in these numbers.

A value for a field the document doesn't have counts as an error, even a
placeholder like "N/A". The confidence gate counts any value as present, so a
placeholder gets past it the same way an invented value does.

## Metrics

| Metric | Meaning |
|---|---|
| Fields correct | Share of all ground-truth fields scored correct, across every document of a supported type |
| Absent fields left empty | Of the deliberately missing fields, how many the model left empty instead of filling in |
| Auto-stored | Documents stored without human review |
| Silent errors | Auto-stored documents with any wrong field, a wrong type, or a missing field that should have sent them to review. These are the costly ones: nobody looks at them. The results page lists each one and what was wrong |
| Auto-store precision | Share of auto-stored documents that were entirely correct |
| Straight-through | Of the documents that should be stored without review, how many were |
| Review catch rate | Of the documents that should go to review, how many did |
| Cost per 1,000 docs | From the API's token counts and the prices in `pricing.py` |
| Errors | Documents where the pipeline raised an exception. Their fields count as not extracted and the tokens they used still count, so a crash lowers the numbers instead of dropping out of them |

Every document is sent under a neutral file name (`document.pdf`), since the
dataset's own file names would tell the classifier the answer.

## Running it

```bash
python -m evals.build_dataset          # only needed after changing the generator
python -m evals.run_eval --model gpt-4o-mini --model gpt-6-luna
python -m evals.run_eval --model gpt-6-luna --reasoning-effort low
```

Each run makes real API calls. A full run over 44 documents costs well under
a dollar on the small models.

## Limits

- **Small and synthetic.** 44 documents from templates the pipeline's author
  wrote. Use the numbers to compare models and changes, not as a production
  accuracy figure. Real documents have layouts, languages and handwriting
  these templates don't.
- **Simulated scans.** Skew, blur and compression applied to a clean render.
  Real bad scans also have shadows, folds, stamps and handwritten notes.
- **One run per configuration.** Model output varies a little from run to
  run, so a difference of one or two documents is within noise.
- **No reviewer in the loop.** Documents that pause for review aren't
  resumed, so corrections aren't measured.
