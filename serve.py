"""
Sage 5.0 — Chat Server

Serves the trained Sage model via HTTP with SSE streaming, multi-turn
conversation, and a web chat UI.

Endpoints:
    GET  /              — Serve web/index.html
    GET  /api/info      — Model metadata
    GET  /api/health    — Health check
    POST /api/generate  — Standard generation
    POST /api/stream    — SSE streaming generation
    POST /api/chat      — Multi-turn conversation

Usage::

    python serve.py                   # Default port 8888
    python serve.py --port 9000       # Custom port
    python serve.py --model my.pt     # Custom checkpoint
"""

__all__ = ["SageServer"]

import argparse
import json
import logging
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import tiktoken
import torch

from sage.config import SageConfig
from sage.generation import extract_assistant_response, top_p_sample
from sage.sage import SageModel

logger = logging.getLogger(__name__)

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Accept",
    "Access-Control-Max-Age": "86400",
}


class SageServer:
    """Encapsulates the model, tokenizer, and device state for serving."""

    def __init__(self, checkpoint_path: str = "sage_chat_model.pt") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: SageModel
        self.encoder: tiktoken.Encoding
        self.config: SageConfig
        self._lock = threading.Lock()
        self._load(checkpoint_path)

    def _load(self, path: str) -> None:
        logger.info("Loading model from %s...", path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        config_dict = checkpoint["config"]
        self.config = SageConfig(**{k: v for k, v in config_dict.items() if k != "name"})
        self.config.name = config_dict.get("name", "sage")

        self.model = SageModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()

        tok_name = checkpoint.get("tokenizer_name", "cl100k_base")
        self.encoder = tiktoken.get_encoding(tok_name)

        total = sum(p.numel() for p in self.model.parameters())
        logger.info("Model: %s params on %s", f"{total:,}", self.device)
        logger.info("Tokenizer: %s (%s tokens)", tok_name, f"{self.encoder.n_vocab:,}")

    def _format_prompt(self, prompt: str) -> str:
        return f"User: {prompt}\nAssistant:"

    def _format_chat(self, messages: list) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> dict:
        full_prompt = self._format_prompt(prompt)
        ids = self.encoder.encode(full_prompt, allowed_special=set())
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        t0 = time.time()
        gen_count = 0

        with self._lock:
            for _ in range(max_tokens):
                if input_ids.shape[1] > self.config.n_active_limit:
                    input_ids = input_ids[:, -self.config.n_active_limit:]

                self.model.metacog.reset()
                out = self.model(input_ids)
                next_token = top_p_sample(out["logits"][:, -1, :], temperature, top_p=0.9)

                input_ids = torch.cat([input_ids, next_token], dim=1)
                gen_count += 1

                tok_id = next_token.item()
                if tok_id == self.encoder.eot_token:
                    break

                if gen_count > 3:
                    tail = self.encoder.decode(input_ids[0, -12:].tolist())
                    if "User:" in tail:
                        break

        elapsed = time.time() - t0
        tok_s = gen_count / max(elapsed, 1e-6)

        full_text = self.encoder.decode(input_ids[0].tolist())
        response = extract_assistant_response(full_text, len(full_prompt))

        return {
            "text": response,
            "tokens_generated": gen_count,
            "time_ms": int(elapsed * 1000),
            "tokens_per_sec": round(tok_s, 1),
            "think_iterations": out["metrics"]["iterations_used"],
        }

    @torch.no_grad()
    def generate_stream(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7):
        full_prompt = self._format_prompt(prompt)
        ids = self.encoder.encode(full_prompt, allowed_special=set())
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        t0 = time.time()
        gen_count = 0
        prev_text_len = 0

        with self._lock:
            for _ in range(max_tokens):
                if input_ids.shape[1] > self.config.n_active_limit:
                    input_ids = input_ids[:, -self.config.n_active_limit:]

                self.model.metacog.reset()
                out = self.model(input_ids)
                next_token = top_p_sample(out["logits"][:, -1, :], temperature, top_p=0.9)

                input_ids = torch.cat([input_ids, next_token], dim=1)
                gen_count += 1

                tok_id = next_token.item()

                full_text = self.encoder.decode(input_ids[0].tolist())
                response = extract_assistant_response(full_text, len(full_prompt))
                new_text = response[prev_text_len:]
                prev_text_len = len(response)

                if tok_id == self.encoder.eot_token:
                    break

                if gen_count > 3:
                    tail = self.encoder.decode(input_ids[0, -12:].tolist())
                    if "User:" in tail:
                        break

                yield {"token": new_text, "done": False}

        elapsed = time.time() - t0
        tok_s = gen_count / max(elapsed, 1e-6)

        yield {
            "token": "",
            "done": True,
            "metrics": {
                "tokens_generated": gen_count,
                "time_ms": int(elapsed * 1000),
                "tokens_per_sec": round(tok_s, 1),
                "think_iterations": out["metrics"]["iterations_used"],
            },
        }

    @torch.no_grad()
    def chat(self, messages: list, max_tokens: int = 150, temperature: float = 0.7) -> dict:
        full_prompt = self._format_chat(messages)
        ids = self.encoder.encode(full_prompt, allowed_special=set())
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        t0 = time.time()
        gen_count = 0

        with self._lock:
            for _ in range(max_tokens):
                if input_ids.shape[1] > self.config.n_active_limit:
                    input_ids = input_ids[:, -self.config.n_active_limit:]

                self.model.metacog.reset()
                out = self.model(input_ids)
                next_token = top_p_sample(out["logits"][:, -1, :], temperature, top_p=0.9)

                input_ids = torch.cat([input_ids, next_token], dim=1)
                gen_count += 1

                tok_id = next_token.item()
                if tok_id == self.encoder.eot_token:
                    break

                if gen_count > 3:
                    tail = self.encoder.decode(input_ids[0, -12:].tolist())
                    if "User:" in tail:
                        break

        elapsed = time.time() - t0
        tok_s = gen_count / max(elapsed, 1e-6)

        full_text = self.encoder.decode(input_ids[0].tolist())
        response = extract_assistant_response(full_text, len(full_prompt))

        return {
            "text": response,
            "tokens_generated": gen_count,
            "time_ms": int(elapsed * 1000),
            "tokens_per_sec": round(tok_s, 1),
            "think_iterations": out["metrics"]["iterations_used"],
        }

    def get_info(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        counts = self.model.count_parameters()
        return {
            "name": self.config.name,
            "total_params": total,
            "active_params": counts["active_per_token"],
            "device": self.device,
            "vocab_size": self.encoder.n_vocab,
            "graph_nodes": self.config.n_nodes,
            "active_limit": self.config.n_active_limit,
            "max_think": self.config.max_think_iterations,
            "is_chat": True,
            "version": "5.0",
            "architecture": "Wave Propagation + Resonance Memory",
        }


class ChatHandler(BaseHTTPRequestHandler):

    @property
    def server_instance(self) -> SageServer:
        return self.server.server_instance

    @property
    def max_tokens_cap(self) -> int:
        return getattr(self.server, "max_tokens_cap", 500)

    def _send_cors(self):
        for k, v in _CORS_HEADERS.items():
            self.send_header(k, v)

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, status: int, message: str) -> None:
        self._json_response({"error": message}, status=status)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        t0 = time.time()
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
            try:
                with open(html_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self._send_cors()
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self._json_error(404, "index.html not found")

        elif path == "/api/info":
            self._json_response(self.server_instance.get_info())

        elif path == "/api/health":
            self._json_response({"status": "ok", "model_loaded": True})

        else:
            self._json_error(404, f"Not found: {path}")

        elapsed_ms = (time.time() - t0) * 1000
        logger.debug("GET %s %d ms", path, elapsed_ms)

    def do_POST(self):
        t0 = time.time()
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
        except (json.JSONDecodeError, ValueError) as e:
            self._json_error(400, f"Invalid JSON: {e}")
            return

        if path == "/api/generate":
            prompt = body.get("prompt", "").strip()
            if not prompt:
                self._json_error(400, "Missing 'prompt' field")
                return
            max_tokens = min(body.get("max_tokens", 150), self.max_tokens_cap)
            temperature = body.get("temperature", 0.7)

            try:
                result = self.server_instance.generate(prompt, max_tokens, temperature)
                self._json_response(result)
            except Exception as e:
                logger.exception("Generation error")
                self._json_error(500, str(e))

        elif path == "/api/stream":
            prompt = body.get("prompt", "").strip()
            if not prompt:
                self._json_error(400, "Missing 'prompt' field")
                return
            max_tokens = min(body.get("max_tokens", 150), self.max_tokens_cap)
            temperature = body.get("temperature", 0.7)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._send_cors()
            self.end_headers()

            try:
                for event in self.server_instance.generate_stream(prompt, max_tokens, temperature):
                    data = json.dumps(event)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except BrokenPipeError:
                logger.debug("Client disconnected during stream")
            except Exception as e:
                logger.exception("Streaming error")
                error_event = json.dumps({"error": str(e), "done": True})
                try:
                    self.wfile.write(f"data: {error_event}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except BrokenPipeError:
                    pass

        elif path == "/api/chat":
            messages = body.get("messages", [])
            if not messages:
                self._json_error(400, "Missing 'messages' field")
                return
            for msg in messages:
                if "role" not in msg or "content" not in msg:
                    self._json_error(400, "Each message must have 'role' and 'content'")
                    return
                if msg["role"] not in ("user", "assistant", "system"):
                    self._json_error(400, f"Invalid role: {msg['role']}")
                    return
            max_tokens = min(body.get("max_tokens", 150), self.max_tokens_cap)
            temperature = body.get("temperature", 0.7)

            try:
                result = self.server_instance.chat(messages, max_tokens, temperature)
                self._json_response(result)
            except Exception as e:
                logger.exception("Chat error")
                self._json_error(500, str(e))

        else:
            self._json_error(404, f"Not found: {path}")

        elapsed_ms = (time.time() - t0) * 1000
        logger.debug("POST %s %d ms", path, elapsed_ms)

    def log_message(self, format, *args):
        logger.debug(format, *args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Sage 5.0 Chat Server")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument("--model", type=str, default="sage_chat_model.pt", help="Checkpoint path")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max generation tokens cap")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error("No trained model found at '%s'. Run: python train_chat.py", args.model)
        return

    server_instance = SageServer(args.model)

    httpd = HTTPServer(("0.0.0.0", args.port), ChatHandler)
    httpd.server_instance = server_instance
    httpd.max_tokens_cap = args.max_tokens

    logger.info("Sage Chat Server running at http://localhost:%d", args.port)
    logger.info("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
