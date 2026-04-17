"""
Qwen3.5-4B interactive chat — mlx-lm con streaming, multi-turn, tools y reasoning.

Uso:
    uv run chat.py

Dependencias (pyproject.toml):
    mlx-lm>=0.21.0
"""

import json
import re

from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

from tools import TOOLS, execute_tool

# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_ID   = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
REASONING  = "low"    # "low" | "medium" | "high"
MAX_TOKENS = 4096
USE_TOOLS  = True

SYSTEM_PROMPT = (
    f"/think_budget {REASONING}. "
    "Eres un asistente útil y conciso. Responde siempre en el idioma del usuario."
)

# ── Regex ─────────────────────────────────────────────────────────────────────

_TOOL_CALL_RE  = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINKING_RE   = re.compile(r"<think>.*?</think>", re.DOTALL)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_tool_calls(text: str) -> list[dict]:
    """Extrae todas las tool calls del texto generado por el modelo."""
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    return calls


def strip_thinking(text: str) -> str:
    """Elimina el bloque <think>…</think> para mostrar solo la respuesta."""
    return _THINKING_RE.sub("", text).strip()


def build_prompt(messages: list[dict], tokenizer, tools: list | None) -> str:
    kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def print_separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def generate_streaming(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    *,
    silent: bool = False,
) -> str:
    """
    Genera tokens con streaming.

    - silent=False  → imprime token a token (respuesta final visible).
    - silent=True   → acumula sin imprimir (pasadas intermedias de tool calling).

    Retorna el texto completo generado.
    """
    full_text = ""

    if not silent:
        print("\nAsistente: ", end="", flush=True)

    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        chunk: str = response.text
        full_text += chunk
        if not silent:
            print(chunk, end="", flush=True)

    if not silent:
        print()  # salto de línea al terminar stream

    return full_text


# ── Bucle principal ───────────────────────────────────────────────────────────

def chat_loop(model, tokenizer) -> None:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    active_tools = TOOLS if USE_TOOLS else None

    print(f"\n🤖  Qwen3.5-4B Chat  •  reasoning: {REASONING}  •  tools: {USE_TOOLS}")
    print('    Escribe "exit" o presiona Ctrl+C para salir.\n')
    print_separator()

    while True:
        # ── Leer input del usuario ────────────────────────────────────────
        try:
            user_input = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nHasta luego.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "salir"}:
            print("Hasta luego.")
            break

        messages.append({"role": "user", "content": user_input})

        # ── Primera pasada ────────────────────────────────────────────────
        # Si tools están activas, la ocultamos: puede contener XML de tool call.
        # Si no hay tools, esta ya es la respuesta final → la mostramos.
        prompt = build_prompt(messages, tokenizer, active_tools)
        is_first_pass_silent = USE_TOOLS  # ocultamos si puede haber tool calls
        raw = generate_streaming(model, tokenizer, prompt, MAX_TOKENS, silent=is_first_pass_silent)

        # ── Ciclo de tool calling ─────────────────────────────────────────
        tool_calls = parse_tool_calls(raw) if USE_TOOLS else []

        while tool_calls:
            messages.append({"role": "assistant", "content": raw})

            for call in tool_calls:
                tool_name: str = call.get("name", "")
                tool_args: dict = call.get("arguments", {})

                print(f"\n⚙  Tool: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

                result = execute_tool(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": result,
                })

            # Segunda pasada: síntesis con resultados → siempre visible
            prompt = build_prompt(messages, tokenizer, active_tools)
            raw = generate_streaming(model, tokenizer, prompt, MAX_TOKENS, silent=False)
            tool_calls = parse_tool_calls(raw)

        # ── Si no hubo tools, mostrar la respuesta acumulada ──────────────
        # (La primera pasada fue silent=True pero no hubo tool calls,
        # así que raw tiene la respuesta final sin haberla imprimido.)
        if USE_TOOLS and not tool_calls:
            visible = strip_thinking(raw)
            print(f"\nAsistente: {visible}")

        # Guardar en historial el texto completo (con <think> si lo hay)
        messages.append({"role": "assistant", "content": raw})

        print_separator()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Cargando modelo…")
    model, tokenizer = load(MODEL_ID)
    make_prompt_cache(model)   # warm-up del cache
    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()