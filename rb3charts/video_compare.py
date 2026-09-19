"""Create a side-by-side chart review workspace from a run folder and video URL."""
from __future__ import annotations

from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from threading import Thread
from urllib.parse import parse_qs, urlparse
import webbrowser


def youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().split(":")[0]
    video_id = ""
    if host in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            video_id = parsed.path.strip("/").split("/")[1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id):
        raise ValueError("Enter a valid YouTube gameplay-video URL")
    return video_id


def resolve_review_folder(value: Path | str) -> tuple[Path, Path]:
    path = Path(value).expanduser().resolve()
    folder = path.parent if path.is_file() and path.name.lower() == "review.html" else path
    review = folder / "review.html"
    if not folder.is_dir() or not review.is_file():
        raise FileNotFoundError(f"No review.html found in {folder}")
    return folder, review


def create_video_comparison(review_input: Path | str, video_url: str,
                            output: Path | str = Path("comparisons"), open_browser: bool = False) -> dict:
    folder, review = resolve_review_folder(review_input)
    video_id = youtube_video_id(video_url)
    destination = Path(output).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = destination / f"{folder.name}_video_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    session = {
        "schema": 1, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "review_folder": str(folder), "review_page": str(review),
        "video_url": video_url, "youtube_video_id": video_id,
        "purpose": "Human visual comparison; reference video data is never used as generation input.",
    }
    session_path = run / "comparison-session.json"
    page_path = run / "video-comparison.html"
    session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    page_path.write_text(_page(session, "/review/review.html"), encoding="utf-8")
    result = {**session, "folder": str(run), "html": str(page_path), "session": str(session_path)}
    if open_browser:
        webbrowser.open(page_path.as_uri())
    return result


def start_comparison_server(result: dict):
    """Serve the workspace and source review so media and embeds share an origin."""
    workspace = Path(result["folder"]).resolve()
    review_root = Path(result["review_folder"]).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            request_path = urlparse(self.path).path
            if request_path.startswith("/review/"):
                root, relative = review_root, request_path[len("/review/"):]
            else:
                root, relative = workspace, request_path.lstrip("/") or "video-comparison.html"
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                self.send_error(403); return
            if not candidate.is_file():
                self.send_error(404); return
            size = candidate.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                if match:
                    if match.group(1): start = int(match.group(1))
                    if match.group(2): end = min(end, int(match.group(2)))
            if start > end or start >= size:
                self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers(); return
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1)); self.send_header("Accept-Ranges", "bytes")
            if range_header: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with candidate.open("rb") as stream:
                stream.seek(start); remaining = end - start + 1
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block: break
                    try: self.wfile.write(block)
                    except (BrokenPipeError, ConnectionResetError): break
                    remaining -= len(block)

        def do_POST(self):
            if urlparse(self.path).path != "/observations.json":
                self.send_error(404); return
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2 * 1024 * 1024:
                self.send_error(413); return
            try:
                payload = json.loads(self.rfile.read(length))
                (workspace / "observations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except (ValueError, OSError) as exc:
                self.send_error(400, str(exc)); return
            self.send_response(204); self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, name="rb3-comparison-server", daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/video-comparison.html"


def _page(session: dict, review_uri: str) -> str:
    session_json = json.dumps(session).replace("</", "<\\/")
    video_id = escape(session["youtube_video_id"], quote=True)
    review_uri = escape(review_uri, quote=True)
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>RB3 chart video comparison</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#0c111b;color:#e8eef8;font:15px system-ui}}header{{padding:14px 20px;background:#151d2b;border-bottom:1px solid #334059}}h1{{font-size:20px;margin:0 0 5px}}.muted{{color:#aeb9cc}}.controls{{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:12px 20px;background:#111827}}button,input,select{{font:inherit;color:#eef4ff;background:#253149;border:1px solid #52627d;border-radius:7px;padding:7px 10px}}button{{cursor:pointer}}input[type=number]{{width:90px}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px;height:62vh}}.pane{{display:flex;flex-direction:column;min-width:0}}.pane h2{{font-size:14px;margin:0;padding:7px 10px;background:#192234}}.pane iframe,.pane>div{{flex:1;min-height:0;width:100%;border:1px solid #334059;background:#080b11}}.review{{padding:12px 20px}}textarea{{width:100%;min-height:54px;background:#151d2b;color:white;border:1px solid #52627d;border-radius:7px;padding:8px}}table{{width:100%;border-collapse:collapse;margin-top:10px}}th,td{{border-bottom:1px solid #303b50;padding:7px;text-align:left}}.tag{{color:#79ddc7}}@media(max-width:900px){{.views{{grid-template-columns:1fr;height:auto}}iframe{{height:52vh}}}}
</style></head><body><header><h1>Guided video comparison</h1><div class='muted'>This mode does not extract chart notes automatically. Review folder: {escape(session['review_folder'])}</div></header>
<div class='controls'><button id='play'>Play both</button><button id='pause'>Pause both</button><button id='back'>−5 sec</button><button id='forward'>+5 sec</button><label>Generated time = video time + <input id='offset' type='number' value='0' step='.05'> sec</label><button id='sync'>Sync now</button><label>Speed <select id='speed'><option>1</option><option>.75</option><option>.5</option></select></label><span id='clock' class='tag'>Loading players…</span></div>
<div class='views'><section class='pane'><h2>Official gameplay reference · <a style='color:#79ddc7' href='{escape(session['video_url'], quote=True)}' target='_blank'>Open on YouTube</a></h2><div id='youtube'></div></section><section class='pane'><h2>Generated RB3 Charts review</h2><iframe id='review' src='{review_uri}'></iframe></section></div>
<section class='review'><h2>Record a difference at the current time</h2><div class='controls' style='padding:0'><button data-kind='looks-good'>Looks good</button><button data-kind='extra-note'>Extra generated note</button><button data-kind='missing-note'>Missing generated note</button><button data-kind='wrong-lane'>Wrong lane/pattern</button><button data-kind='timing'>Timing</button><button data-kind='sustain'>Sustain</button></div><textarea id='note' placeholder='Optional detail, such as “repeating G G R Y pattern breaks here”'></textarea><div class='controls' style='padding:8px 0'><button id='export'>Export observations as JSON</button><button id='clear'>Clear observations</button><span class='muted'>Observations are also saved automatically in this comparison folder.</span></div><table><thead><tr><th>Video</th><th>Generated</th><th>Finding</th><th>Note</th></tr></thead><tbody id='observations'></tbody></table></section>
<script src='https://www.youtube.com/iframe_api'></script><script>
const SESSION={session_json};let player=null,playerError='',observations=JSON.parse(localStorage.getItem('rb3-video-compare:'+SESSION.review_folder+':'+SESSION.youtube_video_id)||'[]');
function generatedAudio(){{try{{return document.getElementById('review').contentDocument.getElementById('audio')}}catch(e){{return null}}}}
function onYouTubeIframeAPIReady(){{player=new YT.Player('youtube',{{height:'100%',width:'100%',videoId:'{video_id}',playerVars:{{playsinline:1}},events:{{onReady:()=>{{document.getElementById('clock').textContent='Ready'}},onError:e=>{{playerError='This upload blocks embedded playback — use the separately opened YouTube tab';document.getElementById('clock').textContent=playerError;const frame=document.getElementById('youtube');if(frame)frame.style.opacity='.18';}}}}}})}}
function offset(){{return parseFloat(document.getElementById('offset').value)||0}}function sync(){{const a=generatedAudio();if(player&&a){{a.currentTime=Math.max(0,player.getCurrentTime()+offset());a.playbackRate=parseFloat(document.getElementById('speed').value)}}}}
document.getElementById('sync').onclick=sync;document.getElementById('play').onclick=()=>{{sync();player&&player.playVideo();const a=generatedAudio();if(a)a.play()}};document.getElementById('pause').onclick=()=>{{player&&player.pauseVideo();const a=generatedAudio();if(a)a.pause()}};
function jump(delta){{if(!player)return;player.seekTo(Math.max(0,player.getCurrentTime()+delta),true);sync()}}document.getElementById('back').onclick=()=>jump(-5);document.getElementById('forward').onclick=()=>jump(5);document.getElementById('speed').onchange=()=>{{const rate=parseFloat(document.getElementById('speed').value);if(player)player.setPlaybackRate(rate);const a=generatedAudio();if(a)a.playbackRate=rate}};
function save(){{localStorage.setItem('rb3-video-compare:'+SESSION.review_folder+':'+SESSION.youtube_video_id,JSON.stringify(observations));fetch('/observations.json',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{...SESSION,offset_seconds:offset(),observations}})}});render()}}function render(){{document.getElementById('observations').innerHTML=observations.map(o=>`<tr><td>${{o.video.toFixed(2)}}s</td><td>${{o.generated.toFixed(2)}}s</td><td>${{o.kind}}</td><td>${{o.note.replace(/[<>&]/g,c=>({{'<':'&lt;','>':'&gt;','&':'&amp;'}}[c]))}}</td></tr>`).join('')}}
document.querySelectorAll('[data-kind]').forEach(button=>button.onclick=()=>{{const video=player?player.getCurrentTime():0;observations.push({{video,generated:Math.max(0,video+offset()),kind:button.dataset.kind,note:document.getElementById('note').value.trim()}});document.getElementById('note').value='';save()}});
document.getElementById('clear').onclick=()=>{{if(confirm('Clear all observations for this comparison?')){{observations=[];save()}}}};document.getElementById('export').onclick=()=>{{const blob=new Blob([JSON.stringify({{...SESSION,offset_seconds:offset(),observations}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='video-comparison-observations.json';a.click();URL.revokeObjectURL(a.href)}};
setInterval(()=>{{if(playerError){{document.getElementById('clock').textContent=playerError;return}}const v=player&&player.getCurrentTime?player.getCurrentTime():0,a=generatedAudio();document.getElementById('clock').textContent=`video ${{v.toFixed(2)}}s · generated ${{a?a.currentTime.toFixed(2):'loading'}}s`}},250);render();
</script></body></html>"""
