"""Chat de consola para el asistente de soporte MineCatalog.

Uso:  python chat.py
Requiere la API corriendo (uvicorn src.api:app --host 0.0.0.0 --port 8000).
Escribí tu pregunta y Enter. Para salir: 'salir'.
"""
import requests

URL = "http://localhost:8000/query"

print("=" * 60)
print(" Asistente de Soporte MineCatalog")
print(" Escribí tu pregunta y presioná Enter.  (para salir: salir)")
print("=" * 60)

while True:
    try:
        pregunta = input("\nTu pregunta > ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if pregunta.lower() in {"salir", "exit", "quit"}:
        break
    if not pregunta:
        continue
    try:
        data = requests.post(URL, json={"pregunta": pregunta}, timeout=120).json()
    except Exception as e:
        print(f"\n  [No pude conectar con la API en {URL}. ¿Está corriendo uvicorn? {e}]")
        continue
    print("\n" + data.get("respuesta", "(sin respuesta)"))
    fuentes = data.get("fuentes") or []
    if fuentes:
        refs = ", ".join(
            f"{f['source']}" + (f" · {f['section']}" if f.get("section") else "")
            for f in fuentes
        )
        print(f"\n  Fuentes: {refs}")

print("\n¡Hasta luego!")
