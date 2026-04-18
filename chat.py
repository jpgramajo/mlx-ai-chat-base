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
    """
    Extrae tool calls del texto generado por Qwen3.5.
    Soporta dos formatos:

    Formato JSON (esperado):
        <tool_call>{"name": "get_time", "arguments": {}}</tool_call>

    Formato XML (real de Qwen3.5):
        <tool_call>
        <function=get_time>
        {"location": "Guatemala"}
        </function>
        </tool_call>
    """
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        inner = match.group(1).strip()

        # Formato JSON
        try:
            calls.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass

        # Formato XML: <function=name> [args_json?] </function>
        xml_match = re.match(
            r"<function=(\w+)>\s*(.*?)\s*</function>",
            inner,
            re.DOTALL,
        )
        if xml_match:
            name = xml_match.group(1)
            args_raw = xml_match.group(2).strip()
            try:
                arguments = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                arguments = {}
            calls.append({"name": name, "arguments": arguments})

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


def generate_streaming(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """
    Genera tokens con streaming, imprime chunk a chunk y retorna el texto completo.
    El caller es responsable de imprimir el header antes de llamar.
    """
    full_text = ""

    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        chunk: str = response.text
        full_text += chunk
        print(chunk, end="", flush=True)

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

        # ── Primera pasada — siempre en streaming ────────────────────────
        # Streameamos desde el primer token. Si el modelo emite tool calls,
        # el XML aparecerá en pantalla brevemente; lo limpiamos con el print
        # posterior. En la gran mayoría de turnos no hay tools y el usuario
        # ve la respuesta en tiempo real.
        prompt = build_prompt(messages, tokenizer, active_tools)
        print("\nAsistente: ", end="", flush=True)
        raw = generate_streaming(model, tokenizer, prompt, MAX_TOKENS)

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

            # Síntesis con resultados → también en streaming
            print("\nAsistente: ", end="", flush=True)
            prompt = build_prompt(messages, tokenizer, active_tools)
            raw = generate_streaming(model, tokenizer, prompt, MAX_TOKENS)
            tool_calls = parse_tool_calls(raw)

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