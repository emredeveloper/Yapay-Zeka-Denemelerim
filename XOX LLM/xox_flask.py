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
# Simple engine (minimax-like heuristics)
# -----------------------------


def evaluate_board_for(player: str, board: Board) -> int:
    """Heuristic: +10 for player win, -10 for opponent win, 0 otherwise."""
    winner = check_winner(board)
    if winner == player:
        return 10
    elif winner is not None and winner != player:
        return -10
    return 0


def engine_best_move(player: str, board: Board) -> Tuple[int, int]:
    """Pick a move that wins if possible, blocks opponent wins, else center, corner, random."""
    moves = available_moves(board)
    opponent = "O" if player == "X" else "X"

    # 1) winning move
    for r, c in moves:
        board[r][c] = player
        if check_winner(board) == player:
            board[r][c] = " "
            return r, c
        board[r][c] = " "

    # 2) block opponent winning move
    for r, c in moves:
        board[r][c] = opponent
        if check_winner(board) == opponent:
            board[r][c] = " "
            return r, c
        board[r][c] = " "

    # 3) prefer center
    if (1, 1) in moves:
        return 1, 1

    # 4) prefer corners
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    avail_corners = [m for m in corners if m in moves]
    if avail_corners:
        return random.choice(avail_corners)

    # 5) pick random
    return random.choice(moves)


def leads_to_immediate_loss(player: str, move: Tuple[int, int], board: Board) -> bool:
    """Return True if making `move` as `player` allows the opponent to win immediately next turn."""
    r, c = move
    if board[r][c] != " ":
        return True
    sim = [row[:] for row in board]
    sim[r][c] = player
    opponent = "O" if player == "X" else "X"
    for orow, ocol in available_moves(sim):
        sim[orow][ocol] = opponent
        if check_winner(sim) == opponent:
            return True
        sim[orow][ocol] = " "
    return False



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
        return f"Error: {e}"


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
    history: Optional[List[Dict[str, int]]] = None,
    previous_rationales: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str]:
    assert player_symbol in ("X", "O")
    moves = available_moves(board)
    board_lines = [" ".join(board[r][c] if board[r][c] != " " else "." for c in range(3)) for r in range(3)]
    board_str = "\n".join(board_lines)
    if analysis_mode:
        system_prompt = (
            f"You are a Tic-Tac-Toe (XOX) strategy analyzer evaluating the best move for '{player_symbol}'.\n"
            "Rules: 3x3 board. Three identical symbols in a row/column/diagonal wins.\n"
            "First give a short rationale (2-3 steps of lookahead), then output ONLY JSON: {\"row\":1-3, \"col\":1-3}.\n"
            "Do not include any non-JSON text — do not return rationale and JSON as separate messages; return a single output and do not add characters outside the JSON."
        )
    else:
        system_prompt = (
            f"You are a Tic-Tac-Toe (XOX) player and you are playing as '{player_symbol}'.\n"
            "Rules: 3x3 board. Three identical symbols in a row/column/diagonal wins.\n"
            "Move format: ONLY valid JSON: {\"row\":1-3, \"col\":1-3}.\n"
            "No explanations/comments/code blocks; only JSON.\nIf multiple equally good options exist, you may choose randomly."
        )
    if nonce:
        system_prompt += f"\n(context-key: {nonce})"
    # Structured state object (so the model can see all information)
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
        "history": history or [],
        "previous_rationales": previous_rationales or [],
    }
    state_json = json.dumps(state, ensure_ascii=False)

    if analysis_mode:
        user_prompt = (
            "Board (empty cells shown as '.'):\n" +
            board_str +
            "\nSTATE_JSON:\n" + state_json +
            "\nEvaluate winning/blocking opportunities; produce a short rationale and return ONLY the JSON move."
        )
    else:
        user_prompt = (
            "Current board (empty cells shown as '.'):\n" +
            board_str +
            "\nSTATE_JSON:\n" + state_json +
            "\nReturn your JSON move: {\"row\":r, \"col\":c}"
        )
    return system_prompt, user_prompt


def parse_model_move(text: str) -> Optional[Tuple[int, int]]:
    """Try to extract a JSON move and optional rationale from the model text.

    Returns a tuple (row, col, rationale) where row/col are 0-based ints and
    rationale is a string (or None) if available. Returns None if no valid move.
    """
    if not text:
        return None

    # Attempt to find a JSON object inside the text
    try:
        # naive approach: find the first {...} that parses
        start = text.find("{")
        while start != -1:
            end = text.find("}", start)
            if end == -1:
                break
            candidate = text[start:end+1]
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "row" in data and "col" in data:
                    r = int(data.get("row")) - 1
                    c = int(data.get("col")) - 1
                    if 0 <= r <= 2 and 0 <= c <= 2:
                        # extract rationale as everything before the JSON object (trimmed)
                        rationale = text[:start].strip() or None
                        return (r, c, rationale)
            except Exception:
                pass
            start = text.find("{", start+1)
    except Exception:
        pass

    # Fallback: look for two digits or a single digit 1-9
    digits = [ch for ch in text if ch.isdigit()]
    if digits:
        try:
            idx = int(digits[0])
            if 1 <= idx <= 9:
                idx -= 1
                return (idx // 3, idx % 3, None)
        except Exception:
            pass
    return None


# -----------------------------
# Streaming Match Runner (SSE)
# -----------------------------


def sse_format(data: Dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def match_stream(base_url: str, api_key: str, model_x: str, model_o: str, temperature: float, top_p: float, randomize_prompt: bool, random_start: bool, analysis_mode: bool, analysis_max_tokens: int, engine_fallback: bool = False):
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
    last_move: Optional[Dict[str, int]] = None
    history: List[Dict[str, int]] = []
    previous_rationales: List[Dict[str, str]] = []

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
        last_mv = last_move
        if move_number > 1:
            # We keep last_move between iterations and pass it explicitly so the model can
            # react to the opponent's most recent move. If last_move is None on later turns
            # that's unexpected but harmless because the full board is still provided.
            pass
        sys_p, usr_p = build_move_request(
            board,
            current_symbol,
            nonce,
            analysis_mode,
            move_number,
            last_mv,
            history,
            previous_rationales,
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
        rationale = None

        # parsed may be a tuple (r, c, rationale) or None
        if not parsed:
            chosen = random.choice(moves)
            invalid = True
        else:
            if isinstance(parsed, tuple) and len(parsed) == 3:
                r, c, rationale = parsed
            elif isinstance(parsed, tuple) and len(parsed) >= 2:
                r, c = parsed[0], parsed[1]
            else:
                chosen = random.choice(moves)
                invalid = True
                r = c = None

            if r is not None and c is not None and (r, c) in moves:
                chosen = (r, c)
                invalid = False
            else:
                chosen = random.choice(moves)
                invalid = True

        # engine fallback: override obviously bad choices
        if engine_fallback:
            try_move = chosen
            if invalid or leads_to_immediate_loss(current_symbol, try_move, board):
                chosen = engine_best_move(current_symbol, board)
                invalid = False

        r, c = chosen
        board[r][c] = current_symbol

        # record this move and optional rationale
        last_move = {"row": r + 1, "col": c + 1}
        history.append({"symbol": current_symbol, "row": r + 1, "col": c + 1})
        if analysis_mode and rationale:
            previous_rationales.append({"move_number": move_number, "symbol": current_symbol, "rationale": rationale})

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

    engine_fallback = request.args.get("engine_fallback", "false").lower() in ("1", "true", "yes")

    generator = match_stream(base_url, api_key, model_x, model_o, temperature, top_p, randomize_prompt, random_start, analysis_mode, analysis_max_tokens, engine_fallback=engine_fallback)
    return Response(stream_with_context(generator), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


