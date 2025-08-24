import os
import json
import random
from typing import List, Optional, Tuple, Dict

from flask import Flask, Response, stream_with_context, render_template, request

try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI_CLIENT = True
except Exception:
    _HAS_OPENAI_CLIENT = False

import urllib.request
import urllib.error


app = Flask(__name__, template_folder="templates", static_folder="static")


# -----------------------------
# Game (Tic-Tac-Toe / XOX) Logic
# -----------------------------

Board = List[List[str]]


def create_empty_board() -> Board:
    return [[" ", " ", " "] for _ in range(3)]


def available_moves(board: Board) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]


def check_winner(board: Board) -> Optional[str]:
    lines = []
    for i in range(3):
        lines.append(board[i])
        lines.append([board[0][i], board[1][i], board[2][i]])
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])
    for line in lines:
        if line[0] != " " and line[0] == line[1] == line[2]:
            return line[0]
    return None


def is_full(board: Board) -> bool:
    return all(cell != " " for row in board for cell in row)


def board_to_text(board: Board) -> str:
    lines = []
    for r in range(3):
        row = [board[r][c] if board[r][c] != " " else "." for c in range(3)]
        lines.append(" | ".join(row))
        if r < 2:
            lines.append("---------")
    return "\n".join(lines)


def render_board_html(board: Board) -> str:
    cells = []
    for r in range(3):
        for c in range(3):
            val = board[r][c]
            css = "cell x" if val == "X" else ("cell o" if val == "O" else "cell")
            cells.append(f'<div class="{css}">{val if val.strip() else "&nbsp;"}</div>')
    grid = "".join(cells)
    return f'<div class="board">{grid}</div>'


# -----------------------------
# LM Studio (OpenAI-compatible) Client
# -----------------------------


def _http_post(url: str, data: Dict, headers: Dict) -> Dict:
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"))
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def list_models(base_url: str, api_key: str) -> List[str]:
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            names = []
            for m in models:
                name = m.get("id") or m.get("name")
                if isinstance(name, str):
                    names.append(name)
            return names
    except Exception:
        return []


def call_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    json_response: bool = False,
) -> str:
    if _HAS_OPENAI_CLIENT:
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "top_p": top_p,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            # Some OpenAI-compatible servers support response_format
            if json_response:
                try:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "tic_tac_toe_move",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "row": {"type": "integer", "minimum": 1, "maximum": 3},
                                    "col": {"type": "integer", "minimum": 1, "maximum": 3},
                                },
                                "required": ["row", "col"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    }
                except Exception:
                    pass
            completion = client.chat.completions.create(
                **kwargs,
            )
            content = completion.choices[0].message.content or ""
            return content.strip()
        except Exception:
            pass

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if json_response:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "tic_tac_toe_move",
                "schema": {
                    "type": "object",
                    "properties": {
                        "row": {"type": "integer", "minimum": 1, "maximum": 3},
                        "col": {"type": "integer", "minimum": 1, "maximum": 3},
                    },
                    "required": ["row", "col"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        data = _http_post(url, payload, headers)
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return (content or "").strip()
    except urllib.error.HTTPError as e:
        return f"HTTPError {e.code}: {e.read().decode('utf-8', errors='ignore')}"
    except Exception as e:
        return f"Hata: {e}"


# -----------------------------
# Prompting and Move Parsing
# -----------------------------


def build_move_request(
    board: Board,
    player_symbol: str,
    nonce: Optional[str] = None,
    analysis_mode: bool = False,
    move_number: int = 1,
    last_move: Optional[Dict[str, int]] = None,
) -> Tuple[str, str]:
    assert player_symbol in ("X", "O")
    moves = available_moves(board)
    board_lines = [" ".join(board[r][c] if board[r][c] != " " else "." for c in range(3)) for r in range(3)]
    board_str = "\n".join(board_lines)
    if analysis_mode:
        system_prompt = (
            f"Bir XOX (Tic-Tac-Toe) strateji analizörüsün ve '{player_symbol}' için en iyi hamleyi değerlendiriyorsun.\n"
            "Kurallar: 3x3 tahta. Yatay/dikey/çapraz 3 aynı sembol kazanır.\n"
            "Önce kısa gerekçe ver (2-3 adım ileri bakış), sonra SADECE JSON ver: {\"row\":1-3, \"col\":1-3}.\n"
            "JSON harici metin içermesin — gerekçe ve JSON'u ayrı iki mesajda değil, tek döndür ama JSON dışına karakter koyma."
        )
    else:
        system_prompt = (
            f"Sen bir XOX (Tic-Tac-Toe) oyuncususun ve '{player_symbol}' oynuyorsun.\n"
            "Kurallar: 3x3 tahta. Yatay/dikey/çapraz 3 aynı sembol kazanır.\n"
            "Hamle formatı: SADECE geçerli JSON: {\"row\":1-3, \"col\":1-3}.\n"
            "Açıklama/yorum/kod bloğu YOK; sadece JSON.\nEşit iyi seçenek varsa rastgele seçebilirsin."
        )
    if nonce:
        system_prompt += f"\n(bağlam-anahtarı: {nonce})"
    # Yapılandırılmış durum nesnesi (modelin tüm bilgiyi görmesi için)
    structured_board = [
        [None if cell == " " else cell for cell in row]
        for row in board
    ]
    state = {
        "player": player_symbol,
        "move_number": move_number,
        "board": structured_board,
        "available_moves": [{"row": r + 1, "col": c + 1} for r, c in moves],
        "last_move": last_move,
    }
    state_json = json.dumps(state, ensure_ascii=False)

    if analysis_mode:
        user_prompt = (
            "Tahta (boş hücreler '.' ile):\n" +
            board_str +
            "\nSTATE_JSON:\n" + state_json +
            "\nKazanma/engelleme fırsatlarını değerlendir; kısa gerekçe üret ve SADECE JSON hamleyi döndür."
        )
    else:
        user_prompt = (
            "Mevcut tahta (boş hücreler '.' ile):\n" +
            board_str +
            "\nSTATE_JSON:\n" + state_json +
            "\nJSON hamleni ver: {\"row\":r, \"col\":c}"
        )
    return system_prompt, user_prompt


def parse_model_move(text: str) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    try:
        data = json.loads(text)
        r = int(data.get("row")) - 1
        c = int(data.get("col")) - 1
        if 0 <= r <= 2 and 0 <= c <= 2:
            return r, c
    except Exception:
        pass
    digits = [ch for ch in text if ch.isdigit()]
    if digits:
        try:
            idx = int(digits[0])
            if 1 <= idx <= 9:
                idx -= 1
                return idx // 3, idx % 3
        except Exception:
            pass
    return None


# -----------------------------
# Streaming Match Runner (SSE)
# -----------------------------


def sse_format(data: Dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def match_stream(base_url: str, api_key: str, model_x: str, model_o: str, temperature: float, top_p: float, randomize_prompt: bool, random_start: bool, analysis_mode: bool, analysis_max_tokens: int):
    board = create_empty_board()
    models_available = list_models(base_url, api_key)
    warnings: List[str] = []
    for name in [model_x, model_o]:
        if models_available and name not in models_available:
            warnings.append(name)

    yield sse_format({
        "type": "start",
        "participants": {"X": model_x, "O": model_o},
        "warnings": warnings,
        "board_html": render_board_html(board),
    })

    current_symbol = random.choice(["X", "O"]) if random_start else "X"
    model_for_symbol = {"X": model_x, "O": model_o}
    move_number = 1
    nonce = f"{random.randint(100000, 999999)}" if randomize_prompt else None

    while True:
        winner = check_winner(board)
        if winner or is_full(board):
            result = "draw" if not winner else winner
            winner_model = None
            if winner == "X":
                winner_model = model_x
            elif winner == "O":
                winner_model = model_o
            yield sse_format({
                "type": "end",
                "result": result,
                "winner_model": winner_model,
                "board_html": render_board_html(board),
            })
            break

        model_name = model_for_symbol[current_symbol]
        last_mv = None
        if move_number > 1:
            # Infer last move from board diff is costly; here we keep it simple: send None for first move
            # and fill on client based on stream if needed. Alternatively maintain history server-side.
            pass
        sys_p, usr_p = build_move_request(
            board,
            current_symbol,
            nonce,
            analysis_mode,
            move_number,
            last_mv,
        )
        raw = call_model(
            model=model_name,
            system_prompt=sys_p,
            user_prompt=usr_p,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=analysis_max_tokens if analysis_mode else None,
            json_response=True,
        )
        parsed = parse_model_move(raw)
        moves = available_moves(board)
        if not parsed or parsed not in moves:
            chosen = random.choice(moves)
            invalid = True
        else:
            chosen = parsed
            invalid = False

        r, c = chosen
        board[r][c] = current_symbol

        yield sse_format({
            "type": "move",
            "move_number": move_number,
            "symbol": current_symbol,
            "model": model_name,
            "row": r + 1,
            "col": c + 1,
            "invalid": invalid,
            "raw": raw,
            "board_html": render_board_html(board),
        })

        move_number += 1
        current_symbol = "O" if current_symbol == "X" else "X"


DEFAULT_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
DEFAULT_MODEL_X = os.getenv("MODEL_X", "google/gemma-3n-e4b")
DEFAULT_MODEL_O = os.getenv("MODEL_O", "qwen/qwen3-4b-2507")


@app.route("/")
def index():
    return render_template(
        "index.html",
        base_url=DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
        model_x=DEFAULT_MODEL_X,
        model_o=DEFAULT_MODEL_O,
    )


@app.route("/stream")
def stream():
    base_url = request.args.get("base_url", DEFAULT_BASE_URL)
    api_key = request.args.get("api_key", DEFAULT_API_KEY)
    model_x = request.args.get("model_x", DEFAULT_MODEL_X)
    model_o = request.args.get("model_o", DEFAULT_MODEL_O)
    try:
        temperature = float(request.args.get("temperature", "0.1"))
    except Exception:
        temperature = 0.1

    try:
        top_p = float(request.args.get("top_p", "1.0"))
    except Exception:
        top_p = 1.0
    randomize_prompt = request.args.get("randomize_prompt", "true").lower() in ("1", "true", "yes")
    random_start = request.args.get("random_start", "true").lower() in ("1", "true", "yes")
    analysis_mode = request.args.get("analysis_mode", "true").lower() in ("1", "true", "yes")
    try:
        analysis_max_tokens = int(request.args.get("analysis_max_tokens", "128"))
    except Exception:
        analysis_max_tokens = 128

    generator = match_stream(base_url, api_key, model_x, model_o, temperature, top_p, randomize_prompt, random_start, analysis_mode, analysis_max_tokens)
    return Response(stream_with_context(generator), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


