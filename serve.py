"""
Sage 4.0 — Chat Server

Serves the trained Sage model via HTTP with a web chat UI.

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
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

import tiktoken
import torch
import torch.nn.functional as F

from sage.config import SageConfig
from sage.generation import extract_assistant_response, top_p_sample
from sage.sage import SageModel

logger = logging.getLogger(__name__)


class SageServer:
    """Encapsulates the model, tokenizer, and device state for serving."""

    def __init__(self, checkpoint_path: str = "sage_chat_model.pt") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: SageModel
        self.encoder: tiktoken.Encoding
        self.config: SageConfig
        self._load(checkpoint_path)

    def _load(self, path: str) -> None:
        """Load model and tokenizer from checkpoint."""
        logger.info("Loading model from %s...", path)

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        config_dict = checkpoint["config"]
        self.config = SageConfig(**{k: v for k, v in config_dict.items() if k != "name"})
        self.config.name = config_dict.get("name", "sage")

        self.model = SageModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        tok_name = checkpoint.get("tokenizer_name", "cl100k_base")
        self.encoder = tiktoken.get_encoding(tok_name)

        total = sum(p.numel() for p in self.model.parameters())
        logger.info("Model: %s params on %s", f"{total:,}", self.device)
        logger.info("Tokenizer: %s (%s tokens)", tok_name, f"{self.encoder.n_vocab:,}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> dict:
        """Generate a response to a user prompt."""
        full_prompt = f"User: {prompt}\nAssistant:"
        ids = self.encoder.encode(full_prompt, allowed_special=set())
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        t0 = time.time()
        gen_count = 0

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

            # Stop at next user turn
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
        """Return model metadata for the UI."""
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
            "version": "4.0",
            "architecture": "Wave Propagation + Resonance Memory",
        }


# Module-level server instance (set in main)
_server: SageServer = None  # type: ignore[assignment]


def _make_handler_class():
    """Create a request handler class with access to the server instance."""

    class ChatHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            elif parsed.path == "/api/info":
                self._json_response(_server.get_info())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/generate":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                result = _server.generate(
                    prompt=body.get("prompt", ""),
                    max_tokens=min(body.get("max_tokens", 150), 500),
                    temperature=body.get("temperature", 0.7),
                )
                self._json_response(result)
            else:
                self.send_error(404)

        def _json_response(self, data: dict) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, format, *args):
            logger.debug(format, *args)

    return ChatHandler


def main() -> None:
    """Entry point for the Sage chat server."""
    global _server

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Sage 4.0 Chat Server")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument("--model", type=str, default="sage_chat_model.pt", help="Checkpoint path")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error("No trained model found at '%s'. Run: python train_chat.py", args.model)
        return

    _server = SageServer(args.model)

    httpd = HTTPServer(("0.0.0.0", args.port), _make_handler_class())
    logger.info("Sage Chat Server running at http://localhost:%d", args.port)
    logger.info("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
