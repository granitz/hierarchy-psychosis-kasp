#!/usr/bin/env python
"""
Redact a rendered notebook (HTML) in place before it is committed.

    python scripts/redact_html.py runs/<tag>/reproduce_figures.executed.html [...]

The notebook prints the data paths it resolved and where it saved tables; those
paths contain the username. Rules, with the placeholders HTML-escaped so that they show in the browser instead of being
read as tags:

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

RULES = [
    (re.compile(rb"/Users/[A-Za-z0-9_.-]+"), b"&lt;HOME&gt;"),
    (re.compile(rb"/data/users/[A-Za-z0-9_.-]+"), b"&lt;SERVER_HOME&gt;"),
    (re.compile(rb"/Volumes/[^/<>\"'\n]+"), b"&lt;VOLUME&gt;"),
    (re.compile(rb"sub-[0-9]+"), b"sub-XXXXX"),
]


def redact(data):
    total = 0
    for pattern, replacement in RULES:
        data, n = pattern.subn(replacement, data)
        total += n
    return data, total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        with open(path, "rb") as fh:
            data, n = redact(fh.read())
        with open(path, "wb") as fh:
            fh.write(data)
        print(f"redacted {path}: {n} replacement(s)")
