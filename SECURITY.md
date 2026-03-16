# Security Policy

## Supported versions

Only the latest commit on `main` is actively maintained.

## Sensitive data

This repository contains **no credentials**.
Secrets (HuggingFace tokens, etc.) must be stored in a local `.env` file that
is listed in `.gitignore` and must never be committed.

If you accidentally commit a token, rotate it immediately at
<https://huggingface.co/settings/tokens>.

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Report privately via GitHub's
[Security Advisories](../../security/advisories/new) feature or by e-mail to
the repository owner.  Include a description of the issue and – where possible
– steps to reproduce it.  You will receive a response within 72 hours.
