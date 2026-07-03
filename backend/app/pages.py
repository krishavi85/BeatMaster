from html import escape
from fastapi.responses import HTMLResponse
from .theme import STYLE


def page(title: str, content: str, refresh: bool = False) -> HTMLResponse:
    reload_tag = '<meta http-equiv="refresh" content="3">' if refresh else ""
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{reload_tag}<title>{escape(title)} · BeatMaster</title><style>{STYLE}</style></head><body><div class="shell"><aside><div class="brand"><b>∿</b> BeatMaster</div><nav><a href="/">Dashboard</a><a href="/#new">New project</a><a href="/docs">API documentation</a><a href="/api/capabilities">Capabilities</a></nav><p class="muted">Real files. Real processing. No demonstration values.</p></aside><main>{content}</main></div></body></html>''')
