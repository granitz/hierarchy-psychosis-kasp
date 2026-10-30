#!/usr/bin/env python
"""
Redact an executed notebook (.html or .ipynb) in place before it is committed.

    python scripts/redact_html.py runs/<tag>/reproduce_figures.executed.html [...]

The notebook prints the data paths it resolved and where it saved tables; those
paths contain the username and the folder layout. The data root (KASP_DATA,
as resolved by kasp.config) becomes <DATA>, the model-fit folders <RESULTS>,
<GROUP_RESULTS> and <DERIVED>, and the repository <REPO>; any other path is
shortened by the rules below. In .html the placeholders are
HTML-escaped so that they show in the browser instead of being read as tags.

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


def _roots():
    """(absolute path, placeholder) for the data root and the repository."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    out = []
    try:
        from kasp import config
        if Path(config.BASE_DIR).resolve() != repo / "data":
            out.append((str(config.BASE_DIR).rstrip("/"), b"DATA"))
        # Model-fit folders may sit anywhere; name them by role, not location.
        for attr, name in [("RESULTS_DIR", b"RESULTS"), ("GROUP_RESULTS_DIR", b"GROUP_RESULTS"),
                           ("DERIVED_DIR", b"DERIVED")]:
            out.append((str(getattr(config, attr)).rstrip("/"), name))
    except Exception as err:                                  # noqa: BLE001
        print(f"note: data root not known ({err}); only generic rules applied")
    out.append((str(repo), b"REPO"))
    return sorted(out, key=lambda t: -len(t[0]))


def redact(data, path=".html"):
    total = 0
    wrap = (lambda t: b"&lt;" + t + b"&gt;") if str(path).endswith(".html") \
        else (lambda t: b"<" + t + b">")
    for root, name in _roots():
        raw = root.encode()
        # also the form an earlier, generic-only redaction left behind
        generic = raw
        for pattern, replacement in rules(path)[:-1]:
            generic = pattern.sub(replacement, generic)
        for form in {raw, generic}:
            data, n = re.subn(re.escape(form) + rb"(?![A-Za-z0-9_-])", wrap(name), data)
            total += n
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
