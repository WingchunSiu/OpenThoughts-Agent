"""Deterministic trajectory annotations adapted from TerminalWorld."""

from __future__ import annotations

import re
from dataclasses import replace

from data.terminal_trajectories.schemas import TrajectoryAudit, TrajectoryRecord

_CHECKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "pii_email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+\b"),
    ),
    (
        "pii_public_ipv4",
        re.compile(
            r"\b(?!127\.)(?!10\.)(?!172\.(?:1[6-9]|2[0-9]|3[01])\.)"
            r"(?!192\.168\.)(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
    ),
    (
        "credential_aws_key",
        re.compile(r"\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"),
    ),
    (
        "credential_private_key",
        re.compile(r"-----BEGIN\s+(?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "credential_github_token",
        re.compile(r"\b(?:ghp|ghs|ghu|ghr|github_pat)_[A-Za-z0-9_]{20,}\b"),
    ),
    ("credential_hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    (
        "credential_assignment",
        re.compile(
            r"(?i)(?:export\s+)?(?:[A-Z_]{3,}(?:_KEY|_TOKEN|_SECRET|_PASSWORD|_PASSWD|_PWD))"
            r"\s*=\s*[\"']?[A-Za-z0-9/+\-_.]{20,}[\"']?"
        ),
    ),
    (
        "destructive_recursive_delete",
        re.compile(
            r"\brm\s+(?:[^\n]*\s)?-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+"
            r"(?:/\s|/\*|~/|~\s|~\*|\*\s)"
        ),
    ),
    (
        "destructive_disk_write",
        re.compile(r"\bdd\b[^\n]*\bof=/dev/(?:sd[a-z]|nvme\d|xvd[a-z]|vd[a-z])\b"),
    ),
)

_TUI_TOOLS = (
    "vim",
    "vi",
    "nvim",
    "nano",
    "emacs",
    "less",
    "more",
    "htop",
    "top",
    "btop",
    "tmux",
    "screen",
    "ranger",
    "ncdu",
)
_TUI_INVOCATION = re.compile(
    r"(?:(?:^|[$#%>]\s+))(?:"
    + "|".join(re.escape(tool) for tool in _TUI_TOOLS)
    + r")(?:\s|$)",
    re.MULTILINE,
)
_COMMAND_LINE = re.compile(r"(?:^|\n)(?:[^\n]*[$#%>]\s+)?[^\s\n]+(?:\s+[^\n]+)?")


def audit_trajectory(record: TrajectoryRecord) -> TrajectoryAudit:
    """Return safety annotations without deciding whether to drop the record."""

    reasons = [label for label, pattern in _CHECKS if pattern.search(record.transcript)]
    if _TUI_INVOCATION.search(record.transcript):
        reasons.append("contains_tui")
    line_count = len(record.transcript.splitlines())
    command_count = len(_COMMAND_LINE.findall(record.transcript))
    if line_count < 2:
        reasons.append("too_short")
    return TrajectoryAudit(
        trajectory_id=record.trajectory_id,
        reasons=tuple(sorted(set(reasons))),
        command_count=command_count,
        line_count=line_count,
    )


def redacted_trajectory(record: TrajectoryRecord) -> TrajectoryRecord:
    """Redact sensitive spans before sending a retained record to a teacher."""

    def redact(text: str) -> str:
        for label, pattern in _CHECKS:
            if label.startswith(("pii_", "credential_")):
                text = pattern.sub(f"<redacted:{label}>", text)
        return text

    transcript = redact(record.transcript)
    metadata = {
        key: redact(value) if isinstance(value, str) else value
        for key, value in record.metadata.items()
    }
    return replace(record, transcript=transcript, metadata=metadata)
