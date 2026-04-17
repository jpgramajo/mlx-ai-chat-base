import json
from datetime import datetime

# ── Definición de tools ───────────────────────────────────────────────────────
# Agrega aquí nuevas tools siguiendo el mismo esquema.

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Returns the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ── Implementación de tools ───────────────────────────────────────────────────
# Agrega aquí la lógica de cada tool nueva.

def execute_tool(name: str, arguments: dict) -> str:
    """Despacha la ejecución de una tool por nombre y retorna el resultado como JSON string."""
    if name == "get_time":
        return json.dumps({"datetime": datetime.now().isoformat()})

    return json.dumps({"error": f"Unknown tool: {name}"})