"""Attribution strings for corroborated claims, built in code.

The writer model copies these verbatim, so which outlets get named is
decided here, not by the model. Outlets are ordered by a hash of the claim
and outlet name: stable across runs, but unrelated to the alphabet or to
feeds.yaml order, so no outlet is systematically the one that gets named.
"""
from __future__ import annotations

import hashlib
import re

from corroborate.models import Claim

MAX_NAMED = 3

_PARENTHETICAL = re.compile(r"\(([^()]+)\)")


def format_attribution(claim: Claim) -> str:
    ordered = sorted(
        claim.confirming_outlets,
        key=lambda outlet: hashlib.sha1(
            f"{claim.anchor_text}|{outlet}".encode()
        ).hexdigest(),
    )
    if len(ordered) <= MAX_NAMED:
        return f"({', '.join(ordered)})"
    return f"({', '.join(ordered[:MAX_NAMED])} and others)"


def unverified_attributions(
    text: str, allowed: set[str], known_outlets: set[str]
) -> list[str]:
    """Parentheticals naming an outlet that are not allowed attributions.

    A sentence stating two claims may join their attributions with a
    semicolon, "(A, B, C; D, E and others)"; each part must be allowed.
    """
    unverified = []
    for match in _PARENTHETICAL.finditer(text):
        inner = match.group(1)
        if not any(outlet in inner for outlet in known_outlets):
            continue
        parts = [f"({part.strip()})" for part in inner.split(";")]
        if not all(part in allowed for part in parts):
            unverified.append(match.group(0))
    return unverified
