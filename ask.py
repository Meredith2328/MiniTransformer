from __future__ import annotations

import argparse
import html
import json
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from typing import Any

from cs336_basics.inference import DEFAULT_RUN_DIR, GenerationBackend, build_generation_spec


HTML_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>$title</title>
  <style>
    :root {
      --bg: #f2efe8;
      --bg-accent: #e7eadf;
      --panel: rgba(255, 253, 248, 0.88);
      --panel-strong: #fffdf8;
      --border: rgba(49, 68, 57, 0.14);
      --text: #233128;
      --muted: #637268;
      --primary: #1f6b52;
      --primary-strong: #14543f;
      --assistant: #eef6f2;
      --user: #1e5e49;
      --user-text: #f6fbf8;
      --shadow: 0 24px 70px rgba(34, 48, 40, 0.10);
      --radius-xl: 28px;
      --radius-lg: 20px;
      --radius-md: 16px;
      --mono: "Consolas", "Courier New", monospace;
      --body: "Trebuchet MS", "Gill Sans", sans-serif;
      --title: "Georgia", "Palatino Linotype", serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: var(--body);
      background:
        radial-gradient(circle at top left, rgba(215, 183, 111, 0.16), transparent 24%),
        radial-gradient(circle at bottom right, rgba(31, 107, 82, 0.12), transparent 28%),
        linear-gradient(160deg, var(--bg) 0%, #f7f5ef 46%, var(--bg-accent) 100%);
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.14) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.14) 1px, transparent 1px);
      background-size: 36px 36px;
      mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.55), transparent);
      pointer-events: none;
    }

    .app {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 24px;
      padding: 28px;
      min-height: 100vh;
      position: relative;
      z-index: 1;
    }

    .sidebar,
    .main {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    .sidebar {
      padding: 24px 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .brand {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .brand h1 {
      margin: 0;
      font-family: var(--title);
      font-size: 32px;
      line-height: 1;
      letter-spacing: -0.03em;
    }

    .brand p,
    .hint,
    .status-line,
    .setting label,
    .meta,
    .empty-note {
      color: var(--muted);
    }

    .brand p,
    .hint,
    .meta,
    .empty-note,
    .status-line,
    .setting label {
      margin: 0;
      line-height: 1.45;
    }

    .status-card,
    .settings-card,
    .tips-card {
      padding: 16px;
      border-radius: var(--radius-lg);
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid rgba(49, 68, 57, 0.10);
    }

    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(31, 107, 82, 0.10);
      color: var(--primary-strong);
      font-size: 13px;
      font-weight: 700;
      width: fit-content;
    }

    .status-chip::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--primary);
      box-shadow: 0 0 0 6px rgba(31, 107, 82, 0.12);
    }

    .mono {
      font-family: var(--mono);
      font-size: 12px;
      word-break: break-all;
    }

    .settings-grid {
      display: grid;
      gap: 14px;
      margin-top: 12px;
    }

    .setting {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .setting input {
      width: 100%;
      border: 1px solid rgba(49, 68, 57, 0.14);
      border-radius: 14px;
      padding: 12px 13px;
      background: rgba(255, 255, 255, 0.8);
      color: var(--text);
      font: inherit;
    }

    .main {
      display: flex;
      flex-direction: column;
      min-height: calc(100vh - 56px);
      overflow: hidden;
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 18px;
      padding: 22px 24px 16px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.48), rgba(255, 255, 255, 0.18));
    }

    .topbar h2 {
      margin: 0;
      font-family: var(--title);
      font-size: 32px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }

    .topbar p {
      margin: 6px 0 0;
      color: var(--muted);
    }

    .topbar-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .button,
    .ghost-button,
    .file-label {
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease, background 120ms ease;
    }

    .button:hover,
    .ghost-button:hover,
    .file-label:hover {
      transform: translateY(-1px);
    }

    .button {
      background: var(--primary);
      color: #f7fcf9;
      font-weight: 700;
    }

    .button:disabled {
      cursor: wait;
      opacity: 0.72;
      transform: none;
    }

    .ghost-button,
    .file-label {
      background: rgba(31, 107, 82, 0.08);
      color: var(--primary-strong);
      font-weight: 700;
    }

    .messages {
      flex: 1;
      padding: 22px 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .message {
      display: flex;
      flex-direction: column;
      gap: 8px;
      animation: fadeUp 180ms ease;
    }

    .message.user {
      align-items: flex-end;
    }

    .bubble {
      max-width: min(860px, 88%);
      padding: 16px 18px;
      border-radius: 22px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
      box-shadow: 0 14px 34px rgba(34, 48, 40, 0.08);
    }

    .message.assistant .bubble {
      background: var(--assistant);
      color: var(--text);
      border-top-left-radius: 8px;
    }

    .message.user .bubble {
      background: var(--user);
      color: var(--user-text);
      border-top-right-radius: 8px;
    }

    .message-meta {
      font-size: 12px;
      color: var(--muted);
      padding: 0 6px;
    }

    .composer {
      padding: 16px 20px 20px;
      border-top: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0.42));
    }

    .composer-card {
      background: var(--panel-strong);
      border: 1px solid rgba(49, 68, 57, 0.12);
      border-radius: 22px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
    }

    textarea {
      width: 100%;
      min-height: 138px;
      resize: vertical;
      border: 0;
      outline: none;
      background: transparent;
      font: inherit;
      color: var(--text);
      line-height: 1.6;
    }

    textarea::placeholder {
      color: #87978e;
    }

    .composer-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .file-input {
      display: none;
    }

    .examples {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 4px;
    }

    .example-chip {
      border: 0;
      border-radius: 999px;
      background: rgba(215, 183, 111, 0.18);
      color: #6e5313;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
    }

    .typing {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .typing span {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: rgba(31, 107, 82, 0.55);
      animation: pulse 900ms infinite ease-in-out;
    }

    .typing span:nth-child(2) { animation-delay: 120ms; }
    .typing span:nth-child(3) { animation-delay: 240ms; }

    @keyframes pulse {
      0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
      40% { opacity: 1; transform: translateY(-2px); }
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 960px) {
      .app {
        grid-template-columns: 1fr;
        padding: 16px;
      }

      .main {
        min-height: 72vh;
      }

      .bubble {
        max-width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="status-chip">Single-turn generation</div>
        <h1>MiniTransformer Ask</h1>
        <p>ChatGPT-style local UI for prompt-based generation. Messages are displayed as a conversation, but each request is generated from the current prompt only.</p>
      </div>

      <section class="status-card">
        <div class="hint">Loaded model</div>
        <div class="status-line mono" id="checkpointPath"></div>
        <div class="status-line" id="statusSummary"></div>
      </section>

      <section class="settings-card">
        <div class="hint">Generation settings</div>
        <div class="settings-grid">
          <div class="setting">
            <label for="maxNewTokens">Max new tokens</label>
            <input id="maxNewTokens" type="number" min="1" step="1">
          </div>
          <div class="setting">
            <label for="temperature">Temperature</label>
            <input id="temperature" type="number" min="0" step="0.05">
          </div>
          <div class="setting">
            <label for="topP">Top-p</label>
            <input id="topP" type="number" min="0.01" max="1" step="0.01">
          </div>
        </div>
      </section>

      <section class="tips-card">
        <div class="hint">Quick tips</div>
        <p class="empty-note">Use the file import button to load a prepared prompt into the composer. The clear button resets the page only; there is no server-side memory.</p>
      </section>
    </aside>

    <main class="main">
      <div class="topbar">
        <div>
          <h2>Prompt and generate</h2>
          <p id="subtitle">Type a prompt, send it, and inspect the generated continuation.</p>
          <div class="examples" id="exampleChips"></div>
        </div>
        <div class="topbar-actions">
          <button class="ghost-button" id="clearButton" type="button">Clear</button>
        </div>
      </div>

      <section class="messages" id="messages">
        <div class="message assistant">
          <div class="bubble">Ready. Paste a prompt, import a text file, then click Send.</div>
          <div class="message-meta">The UI is chat-like, but generation is single-turn.</div>
        </div>
      </section>

      <section class="composer">
        <div class="composer-card">
          <textarea id="promptInput" placeholder="Write a prompt here. Shift+Enter adds a new line."></textarea>
          <div class="composer-actions">
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
              <label class="file-label" for="promptFile">Import prompt file</label>
              <input class="file-input" id="promptFile" type="file" accept=".txt,.md,.json,.prompt">
              <div class="meta">No conversation memory. Each click sends only the current prompt.</div>
            </div>
            <button class="button" id="sendButton" type="button">Send</button>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const APP_STATUS = $status_json;
    const DEFAULTS = $defaults_json;
    const EXAMPLES = $examples_json;

    const checkpointPath = document.getElementById("checkpointPath");
    const statusSummary = document.getElementById("statusSummary");
    const maxNewTokens = document.getElementById("maxNewTokens");
    const temperature = document.getElementById("temperature");
    const topP = document.getElementById("topP");
    const promptInput = document.getElementById("promptInput");
    const promptFile = document.getElementById("promptFile");
    const sendButton = document.getElementById("sendButton");
    const clearButton = document.getElementById("clearButton");
    const messages = document.getElementById("messages");
    const exampleChips = document.getElementById("exampleChips");

    checkpointPath.textContent = APP_STATUS.checkpoint;
    statusSummary.textContent = APP_STATUS.device + " | " + APP_STATUS.tokenizer_mode + " | ctx " + APP_STATUS.model_config.context_length;
    maxNewTokens.value = DEFAULTS.max_new_tokens;
    temperature.value = DEFAULTS.temperature;
    topP.value = DEFAULTS.top_p;

    for (const example of EXAMPLES) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-chip";
      button.textContent = example.label;
      button.addEventListener("click", () => {
        promptInput.value = example.prompt;
        promptInput.focus();
      });
      exampleChips.appendChild(button);
    }

    function appendMessage(role, text, metaText, isTyping = false) {
      const wrapper = document.createElement("div");
      wrapper.className = "message " + role;

      const bubble = document.createElement("div");
      bubble.className = "bubble";
      if (isTyping) {
        const typing = document.createElement("div");
        typing.className = "typing";
        typing.innerHTML = "<span></span><span></span><span></span>";
        bubble.appendChild(typing);
      } else {
        bubble.textContent = text;
      }

      const meta = document.createElement("div");
      meta.className = "message-meta";
      meta.textContent = metaText || "";

      wrapper.appendChild(bubble);
      if (metaText || isTyping) {
        wrapper.appendChild(meta);
      }
      messages.appendChild(wrapper);
      messages.scrollTop = messages.scrollHeight;
      return wrapper;
    }

    function resetMessages() {
      messages.innerHTML = "";
      appendMessage("assistant", "Ready. Paste a prompt, import a text file, then click Send.", "The UI is chat-like, but generation is single-turn.");
    }

    async function sendPrompt() {
      const prompt = promptInput.value.trim();
      if (!prompt) {
        promptInput.focus();
        return;
      }

      const payload = {
        prompt,
        max_new_tokens: Number(maxNewTokens.value),
        temperature: Number(temperature.value),
        top_p: Number(topP.value)
      };

      appendMessage("user", prompt, "Prompt sent");
      promptInput.value = "";
      sendButton.disabled = true;
      const pending = appendMessage("assistant", "", "Generating...", true);

      try {
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        pending.remove();

        if (!response.ok) {
          appendMessage("assistant", "Request failed: " + (data.error || "unknown error"), "Server error");
          return;
        }

        const meta = data.completion_tokens + " completion tokens | " + data.elapsed_seconds.toFixed(2) + "s";
        appendMessage("assistant", data.reply, meta);
      } catch (error) {
        pending.remove();
        appendMessage("assistant", "Request failed: " + error, "Network error");
      } finally {
        sendButton.disabled = false;
        promptInput.focus();
      }
    }

    promptFile.addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) {
        return;
      }
      const text = await file.text();
      promptInput.value = text;
      promptInput.focus();
      promptFile.value = "";
    });

    sendButton.addEventListener("click", sendPrompt);
    clearButton.addEventListener("click", resetMessages);
    promptInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendPrompt();
      }
    });
  </script>
</body>
</html>"""
)


EXAMPLE_PROMPTS = [
    {
        "label": "Story seed",
        "prompt": "Write a short TinyStories-style story about a child finding a lantern in the rain.",
    },
    {
        "label": "Dialogue",
        "prompt": "Write a warm dialogue between a fox and a bird who are learning to share food.",
    },
    {
        "label": "Continuation",
        "prompt": "Once upon a time there was a little robot who wanted to grow flowers.",
    },
]


class AskApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.backend = GenerationBackend(
            build_generation_spec(
                checkpoint=args.checkpoint,
                run_dir=args.run_dir,
                config_json=args.config_json,
                tokenizer_mode=args.tokenizer_mode,
                bpe_vocab_path=args.bpe_vocab_path,
                bpe_merges_path=args.bpe_merges_path,
                special_tokens=args.special_tokens,
                eos_token_id=args.eos_token_id,
                device=args.device,
                strict_load=args.strict_load,
            )
        )
        self.status = self.backend.describe()
        self.page = HTML_TEMPLATE.substitute(
            title=html.escape(args.title),
            status_json=json.dumps(self.status),
            defaults_json=json.dumps(
                {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            ),
            examples_json=json.dumps(EXAMPLE_PROMPTS),
        ).encode("utf-8")

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        max_new_tokens = int(payload.get("max_new_tokens", self.args.max_new_tokens))
        temperature = float(payload.get("temperature", self.args.temperature))
        top_p = float(payload.get("top_p", self.args.top_p))
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0.")
        if temperature < 0:
            raise ValueError("temperature must be >= 0.")
        if not (0 < top_p <= 1):
            raise ValueError("top_p must be in (0, 1].")

        start_time = time.perf_counter()
        result = self.backend.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            completion_only=True,
        )
        elapsed = time.perf_counter() - start_time
        return {
            "reply": result.text,
            "elapsed_seconds": elapsed,
            "prompt_tokens": len(result.prompt_token_ids),
            "completion_tokens": len(result.completion_token_ids),
            "checkpoint": self.status["checkpoint"],
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a local browser UI for prompt-based generation.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config-json", type=Path, default=None)

    parser.add_argument("--tokenizer-mode", choices=("bpe", "gpt2", "none"), default="bpe")
    parser.add_argument("--bpe-vocab-path", type=Path, default=None)
    parser.add_argument("--bpe-merges-path", type=Path, default=None)
    parser.add_argument("--special-tokens", type=str, default=None)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--title", type=str, default="MiniTransformer Ask")
    parser.add_argument("--no-browser", action="store_true")
    return parser


def make_handler(app: AskApplication):
    class AskHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: HTTPStatus) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_page(self) -> None:
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(app.page)))
            self.end_headers()
            self.wfile.write(app.page)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send_page()
                return
            if self.path == "/api/status":
                self._send_json(app.status, HTTPStatus.OK)
                return
            self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/generate":
                self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                if not isinstance(payload, dict):
                    raise ValueError("Request payload must be a JSON object.")
                result = app.generate(payload)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # pragma: no cover - defensive path for UI requests
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            self._send_json(result, HTTPStatus.OK)

    return AskHandler


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    app = AskApplication(args)
    handler_cls = make_handler(app)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)

    url = f"http://{args.host}:{args.port}"
    print(f"MiniTransformer Ask listening on {url}")
    print(f"Loaded checkpoint: {app.status['checkpoint']}")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
