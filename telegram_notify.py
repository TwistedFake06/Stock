"""Telegram credentials + send. Used by Streamlit UI and watchlist_alert.py."""
from __future__ import annotations

import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
SECRETS_TOML = ROOT / ".streamlit" / "secrets.toml"
TG_KEYS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


def _parse_kv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val:
            out[key] = val
    return out


def load_telegram_creds() -> tuple[str, str]:
    """Env → Streamlit secrets → .streamlit/secrets.toml → .env."""
    token = (
        os.environ.get("TELEGRAM_BOT_TOKEN")
        or os.environ.get("TG_BOT_TOKEN")
        or ""
    ).strip()
    chat = (
        os.environ.get("TELEGRAM_CHAT_ID")
        or os.environ.get("TG_CHAT_ID")
        or ""
    ).strip()
    if token and chat:
        return token, chat

    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            if not token:
                token = str(secrets.get("TELEGRAM_BOT_TOKEN") or "").strip()
            if not chat:
                chat = str(secrets.get("TELEGRAM_CHAT_ID") or "").strip()
    except Exception:
        pass

    file_kv = {**_parse_kv_file(SECRETS_TOML), **_parse_kv_file(ENV_PATH)}
    token = token or file_kv.get("TELEGRAM_BOT_TOKEN") or file_kv.get("TG_BOT_TOKEN") or ""
    chat = chat or file_kv.get("TELEGRAM_CHAT_ID") or file_kv.get("TG_CHAT_ID") or ""
    token, chat = token.strip(), str(chat).strip()
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    if chat:
        os.environ["TELEGRAM_CHAT_ID"] = chat
    return token, chat


def telegram_configured() -> bool:
    token, chat = load_telegram_creds()
    return bool(token and chat)


def upsert_dotenv(updates: dict[str, str]) -> None:
    """Create/update keys in project .env (local). Cloud may be read-only."""
    existing: list[str] = []
    if ENV_PATH.is_file():
        existing = ENV_PATH.read_text(encoding="utf-8").splitlines()
    keys_done: set[str] = set()
    out: list[str] = []
    for line in existing:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in updates:
                out.append(f"{k}={updates[k]}")
                keys_done.add(k)
                continue
        out.append(line)
    for k, v in updates.items():
        if k not in keys_done:
            out.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def send_telegram(text: str) -> tuple[bool, str]:
    token, chat = load_telegram_creds()
    if not token or not chat:
        return False, "未配置 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat, "text": text},
            timeout=20,
        )
        if r.status_code == 200:
            return True, "telegram ok"
        return False, f"telegram HTTP {r.status_code}: {r.text[:200]}"
    except Exception as exc:
        return False, f"telegram error: {exc}"
