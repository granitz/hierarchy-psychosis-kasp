#!/usr/bin/env python
"""
Redact an executed notebook (.html or .ipynb) in place before it is committed.

    python scripts/redact_html.py runs/<tag>/reproduce_figures.executed.html [...]

The notebook prints the data paths it resolved and where it saved tables; those
paths contain the username. Rules (in .html the placeholders are HTML-escaped
so that they show in the browser instead of being read as tags):

    /Users/<user>          -> <HOME>
    /data/users/<user>     -> <SERVER_HOME>
    /Volumes/<drive>       -> <VOLUME>       (external drives)
    sub-<digits>           -> sub-XXXXX

Embedded figures are base64, which contains no '-', so the participant rule
cannot touch them; the path rules need a literal '/Users/' or '/data/users/'.
Idempotent. Prints how many replacements were made per file.
"""
import re
import sys

PATTERNS = [
    (re.compile(rb"/Users/[A-Za-z0-9_.-]+"), b"HOME"),
    (re.compile(rb"/data/users/[A-Za-z0-9_.-]+"), b"SERVER_HOME"),
    (re.compile(rb"/Volumes/[^/<>\"'\n\\\\]+"), b"VOLUME"),
]
SUBJECT = (re.compile(rb"sub-[0-9]+"), b"sub-XXXXX")


def rules(path):
    """HTML-escaped placeholders for .html, plain ones for .ipynb (JSON)."""
    wrap = (lambda t: b"&lt;" + t + b"&gt;") if str(path).endswith(".html") \
        else (lambda t: b"<" + t + b">")
    return [(p, wrap(t)) for p, t in PATTERNS] + [SUBJECT]


def redact(data, path=".html"):
    total = 0
    for pattern, replacement in rules(path):
        data, n = pattern.subn(replacement, data)
        total += n
    return data, total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        with open(path, "rb") as fh:
            data, n = redact(fh.read(), path)
        with open(path, "wb") as fh:
            fh.write(data)
        print(f"redacted {path}: {n} replacement(s)")
