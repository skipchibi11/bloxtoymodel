"""BloxToy ビューア用のローカルサーバー。

使い方:
    .venv/bin/python viewer/serve.py
    → ブラウザで http://localhost:8000/viewer/ を開く

- data/ と outputs/ の JSON を一覧表示する
- 画面の入力欄から生成できる（.venv の Python で起動したときのみ）
  python3 viewer/serve.py で起動した場合は、表示だけ使える
"""

import json
import sys
import threading
import time
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["data", "outputs"]
sys.path.insert(0, str(ROOT))

_gen = None
_gen_lock = threading.Lock()  # 生成は1つずつ


def get_generator():
    global _gen
    if _gen is None:
        from src.generate import Generator  # torch が必要
        _gen = Generator()
    return _gen


class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/files":
            files = []
            for src in SOURCES:
                d = ROOT / src
                if d.is_dir():
                    files += [f"{src}/{p.name}" for p in sorted(d.glob("*.json"))]
            return self._json(files)
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/generate":
            return self._json({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            text = str(req.get("text", "")).strip()[:100]
            if not text:
                return self._json({"error": "作りたいものを入力してください"}, 400)
            n = max(1, min(int(req.get("n", 2)), 4))
            temp = max(0.1, min(float(req.get("temp", 0.7)), 1.5))
            guidance = max(1.0, min(float(req.get("guidance", 3.0)), 10.0))
            with _gen_lock:
                try:
                    gen = get_generator()
                except ImportError:
                    return self._json({"error": "生成には .venv/bin/python viewer/serve.py で起動してください"}, 500)
                t0 = time.time()
                saved = gen.generate(text, n=n, temp=temp, guidance=guidance)
            return self._json({
                "files": [f"outputs/{p.name}" for p, _ in saved],
                "blocks": [nb for _, nb in saved],
                "seconds": round(time.time() - t0, 1),
                "step": gen.step,
            })
        except RuntimeError as e:
            return self._json({"error": str(e)}, 500)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"生成に失敗しました: {e}"}, 500)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT)))
    print(f"ビューア: http://localhost:{port}/viewer/  （終了は Ctrl+C）")
    try:
        import torch  # noqa: F401
        print("生成: 使えます（初回の生成はモデルの読み込みで少し時間がかかります）")
    except ImportError:
        print("生成: 使えません（.venv/bin/python viewer/serve.py で起動すると使えます）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
