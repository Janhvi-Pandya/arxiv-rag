"""
space_app.py — the entry point for the deployed app (Hugging Face Space).

It reuses the exact FastAPI app from rag.py and only adds one thing:
opening the Space's home page sends you to /docs, where you can try /ask in the browser.
"""

from fastapi.responses import RedirectResponse

from rag import app


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/docs")
