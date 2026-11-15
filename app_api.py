# =====================================================================
# Kaggle notebook cell — app_api.py (server) ဖိုင်ကို ရေးထုတ်ပေးသည်
#   1) ဒီ cell ကို run ပါ      -> app_api.py ဖိုင် ထွက်လာမည်
#   2) !python app_api.py      -> server စတင်မည်
#
# ngrok token: code ထဲက NGROK_TOKEN ကို သုံးသည်
#   (Kaggle Secret NGROK_AUTHTOKEN ရှိရင် အဲဒါကို ဦးစားပေးသုံးမည်)
#
# သတိ: ဒီဖိုင်ကို ဒီ folder ထဲမှာ run ရင် ဒီဖိုင်ကိုယ်တိုင်ကို server code နဲ့ overwrite လုပ်သွားပါမည်။
# =====================================================================
import os

code_content = r'''import os
import re
import math
import time
import uuid
import shutil
import threading
import traceback
import urllib.parse
import urllib.request
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pyngrok import ngrok

PORT = 7860
OUTPUT_DIR = "/kaggle/working/outputs"
REF_DIR = "/tmp/voxcpm_refs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)


# 🔑 ngrok authtoken (environment variable / Kaggle Secret မရှိရင် ဒါကို သုံးမည်)
NGROK_TOKEN = "3M2PsrjjoADbZH2daFyUpPy1Ebz_4pwp6SLMDvnEoN1xxQEzN"


def get_ngrok_token() -> str:
    """ngrok token: environment variable -> Kaggle Secret -> code ထဲက NGROK_TOKEN"""
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        token = (UserSecretsClient().get_secret("NGROK_AUTHTOKEN") or "").strip()
        if token:
            return token
    except Exception:
        pass
    return NGROK_TOKEN.strip()


# Model load က ကြာသောကြောင့် token ကို အရင်စစ်သည်
if __name__ == "__main__" and not get_ngrok_token():
    raise SystemExit("❌ NGROK_AUTHTOKEN မတွေ့ပါ။ Kaggle Secrets (သို့) environment variable တွင် ထည့်ပါ။")


# =====================================================================
# စာသား ခွဲခြင်း
#   အရင် version က စာသားကို အပိုင်းငယ်များစွာ (ဝါကျအလယ်တွေမှာပါ) ဖြတ်ပြီး
#   အပိုင်းတစ်ခုချင်း သီးခြား generate လုပ်သောကြောင့် ဝါကျအလယ်တွင် အသံရပ်သွားခဲ့သည်။
#   ယခု: ။ ? ! တွင်သာ ဖြတ်သည်၊ ဝါကျတိုများကို ပေါင်းသည်၊ ဝါကျရှည်လွန်းမှသာ ၊ တွင် ခွဲသည်။
# =====================================================================
# ဝါကျအဆုံး: ။ ? ! နှင့် နောက်တွင် space/အဆုံး ပါသော "." (1.5 လို ဂဏန်းကို မဖြတ်ရန်)
SENTENCE_RE = re.compile(r".+?(?:[။?!]+|\.(?=\s|$)|$)")
# ဝါကျအတွင်း ပုဒ်ဖြတ်: ၊ , ; :
CLAUSE_RE = re.compile(r".+?(?:[၊,;:]+|$)")

# chunk တစ်ခုပြီးနောက် ထည့်မည့် pause အမျိုးအစား
PARAGRAPH, SENTENCE, COMMA, WORD = "paragraph", "sentence", "comma", "word"


def has_speech(text: str) -> bool:
    """ပုဒ်ဖြတ်သက်သက် (ဥပမာ "။") ကို model ဆီ မပို့မိစေရန်။"""
    return re.search(r"\w", text) is not None


def split_sentences(text: str) -> List[Tuple[str, str]]:
    """စာသားကို ဝါကျများအဖြစ် ခွဲသည် (ပုဒ်ဖြတ်ကို ဝါကျနှင့်အတူ ထားသည်)။"""
    units = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()]
    for para in paragraphs:
        for line in para.split("\n"):
            line = re.sub(r"\s+", " ", line).strip()
            for s in SENTENCE_RE.findall(line):
                s = s.strip()
                if has_speech(s):
                    units.append((s, SENTENCE))
        if units:
            units[-1] = (units[-1][0], PARAGRAPH)
    return units


def split_balanced(text: str, max_chars: int) -> List[str]:
    """ရှည်လွန်းသော စာကို space နေရာများတွင် အရှည်တူနီးပါး အပိုင်းများ ခွဲသည်
    (နောက်ဆုံးမှာ စကားလုံး ၁-၂ လုံးတည်း သီးခြားကျန်ခဲ့ခြင်း မဖြစ်စေရန်)။"""
    if len(text) <= max_chars:
        return [text]
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return [text]
    n_parts = math.ceil(len(text) / max_chars)
    cuts = []
    for k in range(1, n_parts):
        ideal = len(text) * k / n_parts
        best = min(spaces, key=lambda pos: abs(pos - ideal))
        if not cuts or best > cuts[-1]:
            cuts.append(best)
    parts, prev = [], 0
    for c in cuts:
        parts.append(text[prev:c].strip())
        prev = c + 1
    parts.append(text[prev:].strip())
    return [p for p in parts if p]


def split_long_sentence(sentence: str, max_chars: int) -> List[Tuple[str, Optional[str]]]:
    """max_chars ထက်ရှည်သော ဝါကျကို ၊ နေရာတွင် အရင်ခွဲသည်၊ မရမှ space တွင် ခွဲသည်။
    နောက်ဆုံးအပိုင်း၏ pause ကို None ထားသည် (ဝါကျ၏ pause ဖြင့် အစားထိုးမည်)။"""
    if len(sentence) <= max_chars:
        return [(sentence, None)]
    clauses = [c.strip() for c in CLAUSE_RE.findall(sentence) if c.strip()]
    groups, cur = [], ""
    for c in clauses:
        candidate = (cur + " " + c).strip()
        if cur and len(candidate) > max_chars:
            groups.append(cur)
            cur = c
        else:
            cur = candidate
    if cur:
        groups.append(cur)
    pieces = []
    for g in groups:
        sub = split_balanced(g, max_chars)
        for j, s in enumerate(sub):
            pieces.append((s, COMMA if j == len(sub) - 1 else WORD))
    pieces[-1] = (pieces[-1][0], None)
    return pieces


def build_chunks(text: str, max_chars: int, merge_sentences: bool = True) -> List[Tuple[str, str]]:
    """Model ကို တစ်ကြိမ်ချင်း ပို့မည့် chunk များ + ၎င်းတို့နောက်က pause အမျိုးအစား။"""
    units = []
    for sentence, end_kind in split_sentences(text):
        for piece, kind in split_long_sentence(sentence, max_chars):
            units.append((piece, kind or end_kind))
    if not merge_sentences:
        return units
    chunks = []
    for piece, kind in units:
        if chunks and chunks[-1][1] == SENTENCE and len(chunks[-1][0]) + 1 + len(piece) <= max_chars:
            chunks[-1] = (chunks[-1][0] + " " + piece, kind)
        else:
            chunks.append((piece, kind))
    return chunks


# =====================================================================
# အသံ ဆက်ခြင်း
# =====================================================================
def to_mono_float(wav) -> np.ndarray:
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def trim_silence(wav: np.ndarray, sr: int, rel_db: float = -40.0,
                 keep_head: float = 0.03, keep_tail: float = 0.08) -> np.ndarray:
    """Model က chunk အစ/အဆုံးမှာ ထည့်ပေးတဲ့ အသံတိတ်ကို ဖြတ်သည်။
    (အရင်က model အသံတိတ် + 0.25s ထပ်ပေါင်းမိလို့ 0.5-0.9s ရပ်သွားခဲ့သည်)"""
    frame = max(1, int(sr * 0.01))
    n_frames = len(wav) // frame
    if n_frames == 0:
        return wav
    frames = wav[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    voiced = np.nonzero(rms > rms.max() * (10 ** (rel_db / 20.0)))[0]
    if voiced.size == 0:
        return wav
    start = max(0, int(voiced[0]) * frame - int(sr * keep_head))
    end = min(len(wav), (int(voiced[-1]) + 1) * frame + int(sr * keep_tail))
    return wav[start:end]


def apply_fades(wav: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    """ဆက်တဲ့နေရာမှာ click သံ မထွက်စေရန် အစ/အဆုံး fade."""
    n = min(len(wav) // 2, int(sr * ms / 1000))
    if n <= 0:
        return wav
    wav = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]
    return wav


# =====================================================================
# Model + App
# =====================================================================
print("🔄 Loading VoxCPM model...")
from voxcpm import VoxCPM  # noqa: E402

model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
SAMPLE_RATE = int(getattr(getattr(model, "tts_model", None), "sample_rate", 48000))
print(f"✅ Model loaded successfully! (sample rate: {SAMPLE_RATE} Hz)")

app = FastAPI(title="VoxCPM Asynchronous API")

JOBS = {}
JOBS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()  # GPU ပေါ်မှာ job တစ်ခုချင်းသာ run ရန် (out-of-memory မဖြစ်စေရန်)
FILENAME_RE = re.compile(r"[A-Za-z0-9_\-]+\.wav")


class TTSRequest(BaseModel):
    text_input: str
    prompt_text: str
    audio_path: str
    # ---- Optional (မပို့လည်း default ဖြင့် အလုပ်လုပ်သည်) ----
    cfg_value: float = Field(2.0, ge=0.5, le=5.0)
    inference_timesteps: int = Field(20, ge=4, le=50)
    max_chunk_chars: int = Field(200, ge=60, le=500)  # chunk တစ်ခု၏ အများဆုံး စာလုံးရေ
    merge_sentences: bool = True                      # ဝါကျတိုများကို chunk တစ်ခုထဲ ပေါင်းမည်
    sentence_pause: float = Field(0.40, ge=0.0, le=3.0)   # ။ ? ! နောက်
    paragraph_pause: float = Field(0.80, ge=0.0, le=5.0)  # စာပိုဒ်နောက်
    comma_pause: float = Field(0.20, ge=0.0, le=2.0)      # ၊ နောက် (ဝါကျရှည်ကို ခွဲရမှသာ)
    word_pause: float = Field(0.08, ge=0.0, le=1.0)       # ၊ မရှိလို့ space မှာ ခွဲရရင်


def update_job(name: str, **fields):
    with JOBS_LOCK:
        JOBS.setdefault(name, {}).update(fields)


def get_job(name: str) -> Optional[dict]:
    with JOBS_LOCK:
        job = JOBS.get(name)
        return dict(job) if job else None


def check_filename(filename: str):
    if not FILENAME_RE.fullmatch(filename):
        raise HTTPException(status_code=400, detail="ဖိုင်နာမည် မမှန်ကန်ပါ")


def download_reference(url: str, job_id: str) -> str:
    """Request တစ်ခုချင်းစီအတွက် ကိုယ်ပိုင် ref ဖိုင် (request နှစ်ခု တစ်ပြိုင်နက် ဝင်လည်း မရောတော့)"""
    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
    if ext not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        ext = ".wav"
    local_path = os.path.join(REF_DIR, "ref_" + job_id + ext)
    with urllib.request.urlopen(url, timeout=60) as resp, open(local_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    if os.path.getsize(local_path) == 0:
        os.remove(local_path)
        raise ValueError("ဒေါင်းလုပ်ရလာသော ဖိုင် အလွတ်ဖြစ်နေသည်")
    return local_path


def generate_audio_task(filename: str, req: TTSRequest, ref_path: str,
                        chunks: List[Tuple[str, str]], is_temp_ref: bool):
    final_path = os.path.join(OUTPUT_DIR, filename)
    partial_path = final_path[:-4] + ".partial.wav"
    pauses = {
        PARAGRAPH: req.paragraph_pause,
        SENTENCE: req.sentence_pause,
        COMMA: req.comma_pause,
        WORD: req.word_pause,
    }
    try:
        if GPU_LOCK.locked():
            print(f"⏳ တန်းစီစောင့်နေသည်: {filename}")
        with GPU_LOCK:
            start_time = time.time()
            update_job(filename, status="processing", started_at=datetime.now().isoformat(timespec="seconds"))
            print(f"🔄 Background ဖြင့် အသံစတင်ထုတ်နေပါပြီ ({filename}) — chunk {len(chunks)} ခု")
            torch.cuda.empty_cache()

            pieces = [np.zeros(int(SAMPLE_RATE * 0.10), dtype=np.float32)]
            for i, (chunk_text, pause_kind) in enumerate(chunks):
                print(f"   [{i + 1}/{len(chunks)}] ({len(chunk_text)} chars) {chunk_text[:50]}...")
                with torch.no_grad():
                    wav = model.generate(
                        text=chunk_text + " ",
                        prompt_wav_path=ref_path,
                        prompt_text=req.prompt_text.strip(),
                        cfg_value=req.cfg_value,
                        inference_timesteps=req.inference_timesteps,
                    )
                pieces.append(apply_fades(trim_silence(to_mono_float(wav), SAMPLE_RATE), SAMPLE_RATE))
                if i < len(chunks) - 1:
                    pieces.append(np.zeros(int(SAMPLE_RATE * pauses[pause_kind]), dtype=np.float32))
                update_job(filename, done_chunks=i + 1)
                torch.cuda.empty_cache()
            pieces.append(np.zeros(int(SAMPLE_RATE * 0.30), dtype=np.float32))

            final_wav = np.concatenate(pieces)
            peak = float(np.max(np.abs(final_wav))) if final_wav.size else 0.0
            if peak > 0.99:
                final_wav = final_wav * (0.99 / peak)
            # .partial.wav နဲ့ အရင်ရေးပြီးမှ နာမည်ပြောင်း -> ရေးနေတုန်း download လုပ်ရင် ဖိုင်တစ်ဝက် မရတော့
            sf.write(partial_path, final_wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            os.replace(partial_path, final_path)

            elapsed = time.time() - start_time
            update_job(filename, status="done", elapsed_seconds=round(elapsed, 2),
                       audio_seconds=round(len(final_wav) / SAMPLE_RATE, 2))
            print(f"✅ အသံဖိုင် ပြီးဆုံးပါပြီ: {filename} (ကြာချိန်: {elapsed:.2f}s)")
    except Exception as e:
        traceback.print_exc()
        update_job(filename, status="error", error=str(e))
        print(f"❌ Error in background task ({filename}): {e}")
        if os.path.exists(partial_path):
            os.remove(partial_path)
    finally:
        if is_temp_ref and os.path.exists(ref_path):
            os.remove(ref_path)


@app.get("/")
def health():
    return {"ok": True, "model": "openbmb/VoxCPM2", "busy": GPU_LOCK.locked()}


@app.post("/api/v1/generate")
def create_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    if not request.text_input.strip():
        raise HTTPException(status_code=400, detail="စာသား ထည့်ပေးပါ")
    if not request.prompt_text.strip():
        raise HTTPException(status_code=400, detail="prompt_text (reference အသံထဲက စာသား) ထည့်ပေးပါ")

    chunks = build_chunks(request.text_input, request.max_chunk_chars, request.merge_sentences)
    if not chunks:
        raise HTTPException(status_code=400, detail="အသံထုတ်လို့ရမည့် စာသား မတွေ့ပါ")

    job_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "long_clone_" + ts + "_" + job_id + ".wav"

    is_temp_ref = request.audio_path.startswith(("http://", "https://"))
    if is_temp_ref:
        try:
            ref_path = download_reference(request.audio_path, job_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cloudinary လင်ခ်မှ အသံဖိုင် ဒေါင်းလုပ်ဆွဲမရပါ: {e}")
    else:
        ref_path = request.audio_path
        if not os.path.exists(ref_path):
            raise HTTPException(status_code=400, detail="Reference အသံဖိုင် ရှာမတွေ့ပါ။")

    estimated_time = round(len(request.text_input) * 0.12, 2)
    update_job(filename, status="queued", total_chunks=len(chunks), done_chunks=0, error=None,
               created_at=datetime.now().isoformat(timespec="seconds"))
    background_tasks.add_task(generate_audio_task, filename, request, ref_path, chunks, is_temp_ref)

    # 🚀 ဖိုင်နာမည်ကို ချက်ချင်း Return ပြန်မည် (Time-out လုံးဝမရှိပါ)
    return {
        "status": "processing",
        "file_name": filename,
        "estimated_duration_seconds": estimated_time,
        "total_chunks": len(chunks),
        "status_url": "/api/v1/status/" + filename,
        "download_url": "/api/v1/download/" + filename,
    }


@app.get("/api/v1/status/{filename}")
def job_status(filename: str):
    check_filename(filename)
    job = get_job(filename)
    if job is None:
        if os.path.exists(os.path.join(OUTPUT_DIR, filename)):
            return {"file_name": filename, "status": "done"}
        raise HTTPException(status_code=404, detail="ဒီဖိုင်နာမည်နဲ့ job မရှိပါ")
    return {"file_name": filename, **job}


@app.get("/api/v1/download/{filename}")
def get_audio(filename: str):
    check_filename(filename)
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    job = get_job(filename)
    if job and job.get("status") == "error":
        err = job.get("error")
        raise HTTPException(status_code=500, detail=f"အသံထုတ်ရာတွင် error ဖြစ်ခဲ့ပါသည်: {err}")
    if job:
        done, total = job.get("done_chunks", 0), job.get("total_chunks", 0)
        raise HTTPException(status_code=404, detail=f"အသံထုတ်လုပ်နေဆဲ ဖြစ်ပါသည် ({done}/{total} အပိုင်း)")
    raise HTTPException(status_code=404, detail="ဖိုင် ရှာမတွေ့သေးပါ (သို့မဟုတ် အသံထုတ်လုပ်နေဆဲ ဖြစ်ပါသည်)")


if __name__ == "__main__":
    ngrok.set_auth_token(get_ngrok_token())
    tunnel = ngrok.connect(PORT)
    print(f"\n🌍 PUBLIC API URL: {tunnel.public_url}\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
'''

with open("app_api.py", "w", encoding="utf-8") as f:
    f.write(code_content)
print("✅ app_api.py ကို ရေးပြီးပါပြီ။")

# Kaggle session အသစ်တိုင်း pip package တွေ ပျောက်သွားသောကြောင့် မရှိတာကိုသာ install လုပ်သည်
import importlib.util
import subprocess
import sys

REQUIRED = {"pyngrok": "pyngrok", "fastapi": "fastapi", "uvicorn": "uvicorn",
            "soundfile": "soundfile", "voxcpm": "voxcpm"}  # import name -> pip name
missing = [pip_name for mod, pip_name in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("📦 Installing: " + ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
print("✅ လိုအပ်သော package များ အဆင်သင့်ဖြစ်ပါပြီ။")

# `!python app_api.py` (subprocess) က token ကို မြင်နိုင်ရန် Kaggle Secret မှ environment ထဲ ထည့်ထားသည်
if not os.environ.get("NGROK_AUTHTOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["NGROK_AUTHTOKEN"] = UserSecretsClient().get_secret("NGROK_AUTHTOKEN")
        print("✅ Kaggle Secret မှ NGROK_AUTHTOKEN ကို ယူပြီးပါပြီ။")
    except Exception:
        print("ℹ️ Kaggle Secret မရှိပါ — code ထဲက NGROK_TOKEN ကို သုံးပါမည်။")
