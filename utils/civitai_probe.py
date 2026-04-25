# Copyright 2026 kinorax
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request


CIVITAI_BY_HASH_BASE_URL = "https://civitai.com/api/v1/model-versions/by-hash/"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class ProbeProfile:
    name: str
    headers: Mapping[str, str]
    token_mode: str = "none"  # one of: none, header, query


def build_probe_profiles() -> list[ProbeProfile]:
    return [
        ProbeProfile(
            name="accept_only",
            headers={
                "accept": "application/json",
            },
            token_mode="none",
        ),
        ProbeProfile(
            name="browser_ua",
            headers={
                "accept": "application/json",
                "User-Agent": DEFAULT_BROWSER_USER_AGENT,
            },
            token_mode="none",
        ),
        ProbeProfile(
            name="browser_headers",
            headers={
                "accept": "application/json",
                "User-Agent": DEFAULT_BROWSER_USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://civitai.com/",
                "Origin": "https://civitai.com",
            },
            token_mode="none",
        ),
        ProbeProfile(
            name="browser_headers_with_bearer",
            headers={
                "accept": "application/json",
                "User-Agent": DEFAULT_BROWSER_USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://civitai.com/",
                "Origin": "https://civitai.com",
            },
            token_mode="header",
        ),
        ProbeProfile(
            name="browser_headers_with_query_token",
            headers={
                "accept": "application/json",
                "User-Agent": DEFAULT_BROWSER_USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://civitai.com/",
                "Origin": "https://civitai.com",
            },
            token_mode="query",
        ),
    ]


def build_by_hash_url(
    sha256: str,
    *,
    api_key: str | None = None,
    token_mode: str = "none",
) -> str:
    digest = str(sha256 or "").strip().upper()
    if not digest:
        raise ValueError("sha256 is required")

    base = CIVITAI_BY_HASH_BASE_URL + digest
    key = str(api_key or "").strip()

    if token_mode == "query" and key:
        query = url_parse.urlencode({"token": key})
        return f"{base}?{query}"
    return base


def build_request_headers(
    profile_headers: Mapping[str, str],
    *,
    api_key: str | None = None,
    token_mode: str = "none",
) -> dict[str, str]:
    headers = {str(k): str(v) for k, v in profile_headers.items()}
    key = str(api_key or "").strip()
    if token_mode == "header" and key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def probe_by_hash_once(
    sha256: str,
    profile: ProbeProfile,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    api_key: str | None = None,
) -> dict[str, Any]:
    url = build_by_hash_url(sha256, api_key=api_key, token_mode=profile.token_mode)
    headers = build_request_headers(profile.headers, api_key=api_key, token_mode=profile.token_mode)

    req = url_request.Request(url, headers=headers, method="GET")
    try:
        with url_request.urlopen(req, timeout=float(timeout_seconds)) as response:
            status = int(getattr(response, "status", 200))
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                return {
                    "ok": False,
                    "profile": profile.name,
                    "status": status,
                    "url": url,
                    "error": "invalid_payload_type",
                }
            return {
                "ok": status == 200,
                "profile": profile.name,
                "status": status,
                "url": url,
                "modelVersionId": payload.get("id"),
                "modelId": payload.get("modelId"),
                "fileCount": len(payload.get("files") or []),
                "payload": dict(payload),
            }
    except url_error.HTTPError as exc:
        body_preview = ""
        try:
            body_preview = exc.read().decode("utf-8", errors="replace")[:240]
        except Exception:
            body_preview = ""
        return {
            "ok": False,
            "profile": profile.name,
            "status": int(exc.code),
            "url": url,
            "error": "http_error",
            "bodyPreview": body_preview,
        }
    except url_error.URLError as exc:
        return {
            "ok": False,
            "profile": profile.name,
            "status": None,
            "url": url,
            "error": "url_error",
            "reason": str(exc.reason),
        }
    except Exception as exc:
        return {
            "ok": False,
            "profile": profile.name,
            "status": None,
            "url": url,
            "error": "exception",
            "reason": str(exc),
        }


def run_probe(
    sha256: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for profile in build_probe_profiles():
        result = probe_by_hash_once(
            sha256,
            profile,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )
        results.append(result)
    return results


def _print_result_line(result: Mapping[str, Any]) -> None:
    profile = result.get("profile")
    status = result.get("status")
    ok = bool(result.get("ok"))
    if ok:
        model_version_id = result.get("modelVersionId")
        model_id = result.get("modelId")
        file_count = result.get("fileCount")
        print(
            f"[OK] profile={profile} status={status} "
            f"modelVersionId={model_version_id} modelId={model_id} fileCount={file_count}"
        )
        return
    error = result.get("error")
    reason = result.get("reason")
    body_preview = result.get("bodyPreview")
    detail = reason if reason else body_preview
    print(f"[NG] profile={profile} status={status} error={error} detail={detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe Civitai by-hash API with multiple header profiles.")
    parser.add_argument("--sha256", required=True, help="Target SHA256 hash")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout seconds")
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key. If omitted, CIVITAI_API_KEY env var is used.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path for full probe results.",
    )
    args = parser.parse_args(argv)

    api_key = str(args.api_key or "").strip() or str(os.getenv("CIVITAI_API_KEY") or "").strip()
    results = run_probe(
        str(args.sha256),
        timeout_seconds=float(args.timeout),
        api_key=api_key if api_key else None,
    )

    for item in results:
        _print_result_line(item)

    if args.output:
        with open(str(args.output), "w", encoding="utf-8", newline="\n") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote results: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
