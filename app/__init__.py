"""Financial Document Management API with semantic RAG."""

__version__ = "0.1.0"


def _ensure_ca_bundle() -> None:
    """Pick the broadest available CA bundle for HTTPS clients.

    Why this exists:
    - `requests` / `httpx` / `huggingface_hub` default to certifi's bundle.
    - certifi ships only public root CAs.
    - On networks with a TLS-inspection proxy (e.g. Cisco Umbrella / Zscaler)
      or any corporate root, the proxy's root is installed in the OS / brew
      trust store but is NOT in certifi — every HuggingFace download fails
      with `CERTIFICATE_VERIFY_FAILED`.
    - Python's own SSL stack already knows where the OS bundle lives
      (`ssl.get_default_verify_paths().cafile`). We propagate that path to
      `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` so `requests` uses the same
      (broader) trust store as the rest of the system.

    Order of preference (first usable wins):
      1. Operator-provided env var (`REQUESTS_CA_BUNDLE` or `SSL_CERT_FILE`) —
         we never override an explicit choice.
      2. OS default CA file from OpenSSL (`ssl.get_default_verify_paths()`).
      3. certifi bundle as last-resort fallback.
    """
    import os
    import ssl

    if os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE"):
        return  # operator override — respect it.

    candidates: list[str] = []
    try:
        os_cafile = ssl.get_default_verify_paths().cafile
        if os_cafile:
            candidates.append(os_cafile)
    except Exception:
        pass

    try:
        import certifi
        candidates.append(certifi.where())
    except ImportError:
        pass

    for path in candidates:
        if path and os.path.isfile(path):
            for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
                os.environ.setdefault(var, path)
            return


_ensure_ca_bundle()
