"""プロジェクト全体の設定値。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # BloxToy JSON 素材（手を加えない）
LABELS_FILE = ROOT / "labels.json"  # 追加ラベル（日本語名など、任意）
MATERIALS_FILE = ROOT / "materials.json"  # 素材の一覧（自動生成・追加のみ）
PROCESSED_DIR = ROOT / "processed"  # 前処理の結果
OUTPUTS_DIR = ROOT / "outputs"      # 生成結果・プレビュー

GRID = 16            # 16×16×16
COLOR_LEVELS = 16    # RGB 各チャンネルを 16 段階に丸める
COLOR_STEP = 255 // (COLOR_LEVELS - 1)  # = 17
MATERIAL_SLOTS = 64  # 素材の枠（0 = none）。増やすとモデルの作り直しが必要

# テキストエンコーダ（学習しない・固定）。
# revision を固定して、追加学習のたびに同じベクトルが出るようにする。
# 2026-11-14: e5-small と比較した結果、意味の区別がはっきりしている mpnet を採用（README 参照）
TEXT_ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TEXT_ENCODER_REVISION = "4328cf26390c98c5e3c738b4460a05b95f4911f5"  # Apache 2.0（2026-11-14 確認）
TEXT_PREFIX = ""  # このモデルは接頭辞不要
