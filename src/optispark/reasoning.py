"""
OptiSpark Reasoning Engine — HTTP client that proxies to the OptiSpark backend API.
All LLM logic (system prompts, context injection, model selection) lives server-side.
The client never touches the AI API directly.
"""

import json
import os
import requests


def _safe_detail(resp: requests.Response) -> str:
    """Extract error detail from response, falling back to raw text if not JSON."""
    try:
        return resp.json().get("detail", resp.text)
    except (ValueError, KeyError):
        return resp.text


# Server URL resolution order:
# 1. Explicit server_url passed to constructor
# 2. OPTISPARK_SERVER_URL environment variable
# 3. Default production URL below
DEFAULT_SERVER_URL = os.environ.get(
    "OPTISPARK_SERVER_URL",
    "https://optispark-api.onrender.com"  # Production
)


class _RemoteChatSession:
    """Lightweight wrapper that mimics a chat session over HTTP."""

    def __init__(self, session_id: str, server_url: str):
        self.session_id = session_id
        self.server_url = server_url

    def send_message(self, message: str):
        """Send a message and return an object with a .text attribute."""
        resp = requests.post(
            f"{self.server_url}/api/v1/chat/message",
            json={"session_id": self.session_id, "message": message},
            timeout=120,
        )
        if resp.status_code != 200:
            detail = _safe_detail(resp)
            raise RuntimeError(f"Backend error ({resp.status_code}): {detail}")

        class _Response:
            def __init__(self, text):
                self.text = text

        return _Response(resp.json()["text"])


class ReasoningEngine:
    """HTTP-based reasoning engine that delegates to the OptiSpark backend."""

    def __init__(self, server_url=None):
        self.server_url = (server_url or DEFAULT_SERVER_URL).rstrip("/")
        self.model_id = None

    # ───────────────────────────────────────────────────────────────────
    # v0.2.0 — Multi-Turn Chat Session (via backend)
    # ───────────────────────────────────────────────────────────────────

    def start_chat(self, combined_context):
        """Start a chat session on the backend, returns a RemoteChatSession."""
        import time as _time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.server_url}/api/v1/chat/start",
                    json={"combined_context": combined_context},
                    timeout=180,
                )
                break
            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"\n  ⚠ Server waking up... retrying in {wait}s ({attempt + 1}/{max_retries})")
                    _time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Could not connect to OptiSpark backend at {self.server_url}. "
                        f"Is the server running?"
                    )

        if resp.status_code != 200:
            detail = _safe_detail(resp)
            raise RuntimeError(f"Backend error ({resp.status_code}): {detail}")

        data = resp.json()
        self.model_id = data.get("model_used", "unknown")
        return _RemoteChatSession(
            session_id=data["session_id"],
            server_url=self.server_url,
        )

    # ───────────────────────────────────────────────────────────────────
    # v0.1.0 — Single-Turn Generation (backwards compatible)
    # ───────────────────────────────────────────────────────────────────

    def diagnose(self, features):
        """Generate a single-turn diagnosis from DAG features."""
        prompt = (
            f"Analyze these Spark DAG metrics and identify performance bottlenecks:\n"
            f"{json.dumps(features, indent=2)}\n\n"
            f"Provide a concise diagnosis with specific stage references."
        )
        return self._generate(prompt)

    def generate_fix(self, features):
        """Generate optimized PySpark code to fix detected bottlenecks."""
        prompt = (
            f"Given these Spark DAG metrics:\n{json.dumps(features, indent=2)}\n\n"
            f"Generate exact PySpark code to fix the detected bottlenecks. "
            f"Return ONLY the code, no explanation."
        )
        return self._generate(prompt, use_fallback=True)

    # ───────────────────────────────────────────────────────────────────
    # Private Helpers
    # ───────────────────────────────────────────────────────────────────

    def _generate(self, prompt, use_fallback=False):
        """Single-turn generation via the backend."""
        try:
            resp = requests.post(
                f"{self.server_url}/api/v1/generate",
                json={"prompt": prompt, "use_fallback": use_fallback},
                timeout=120,
            )
        except requests.exceptions.RequestException:
            raise RuntimeError(
                f"Could not connect to OptiSpark backend at {self.server_url}. "
                f"Is the server running?"
            )

        if resp.status_code != 200:
            detail = _safe_detail(resp)
            raise RuntimeError(f"Backend error ({resp.status_code}): {detail}")

        data = resp.json()
        self.model_id = data.get("model_used", "unknown")
        return data["text"]