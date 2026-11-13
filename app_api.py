code_content = '''import os
import torch
import soundfile as sf
import numpy as np
import shutil
import re
from datetime import datetime
import time
import urllib.request
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok

app = FastAPI(title="VoxCPM Asynchronous API")

print("🔄 Loading VoxCPM model...")
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
print("✅ Model loaded successfully!")

OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("/tmp", exist_ok=True)

class TTSRequest(BaseModel):
    text_input: str
    prompt_text: str
    audio_path: str

def generate_audio_task(text_input: str, audio_path: str, prompt_text: str, filename: str):
    start_time = time.time()
    try:
        torch.cuda.empty_cache()
        sentences = re.split(r'([။\\.?!\\n]+)', text_input.strip())
        target_chunks = [s.strip() for s in sentences if s.strip()]
        if not target_chunks:
            target_chunks = [text_input.strip()]

        sample_rate = getattr(getattr(model, "tts_model", None), "sample_rate", 48000)
        silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
        all_wavs = []

        print(f"🔄 Background ဖြင့် အသံစတင်ထုတ်နေပါပြီ ({filename})...")
        for chunk in target_chunks:
            with torch.no_grad():
                wav_chunk = model.generate(
                    text=chunk + " ",
                    prompt_wav_path=audio_path,
                    prompt_text=prompt_text.strip(),
                    cfg_value=2.0,
                    inference_timesteps=20
                )
                if isinstance(wav_chunk, torch.Tensor):
                    wav_chunk = wav_chunk.detach().cpu().numpy()
                if wav_chunk.ndim > 1:
                    wav_chunk = wav_chunk.squeeze()
            all_wavs.append(wav_chunk)
            all_wavs.append(silence)
            torch.cuda.empty_cache()

        final_wav_np = np.concatenate(all_wavs)
        file_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(file_path, final_wav_np, sample_rate)
        
        elapsed = time.time() - start_time
        print(f"✅ အသံဖိုင် ပြီးဆုံးပါပြီ: {filename} (ကြာချိန်: {elapsed:.2f}s)")
    except Exception as e:
        print(f"❌ Error in background task: {str(e)}")

@app.post("/api/v1/generate")
def create_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    if not request.text_input.strip():
        raise HTTPException(status_code=400, detail="စာသား ထည့်ပေးပါ")
    
    local_audio_path = request.audio_path
    if request.audio_path.startswith("http"):
        try:
            local_audio_path = "/tmp/ref_audio.wav"
            urllib.request.urlretrieve(request.audio_path, local_audio_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cloudinary လင်ခ်မှ အသံဖိုင် ဒေါင်းလုပ်ဆွဲမရပါ: {str(e)}")

    if not os.path.exists(local_audio_path):
        raise HTTPException(status_code=400, detail="Reference အသံဖိုင် ရှာမတွေ့ပါ။")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"long_clone_{ts}.wav"
    estimated_time = round(len(request.text_input) * 0.12, 2)

    background_tasks.add_task(generate_audio_task, request.text_input, local_audio_path, request.prompt_text, filename)

    # 🚀 ဖိုင်နာမည်ကို ချက်ချင်း Return ပြန်မည် (Time-out လုံးဝမရှိပါ)
    return {
        "status": "processing",
        "file_name": filename,
        "estimated_duration_seconds": estimated_time
    }

@app.get("/api/v1/download/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="ဖိုင် ရှာမတွေ့သေးပါ (သို့မဟုတ် အသံထုတ်လုပ်နေဆဲ ဖြစ်ပါသည်)")

if __name__ == "__main__":
    public_url = ngrok.connect(7860)
    print(f"\\n🌍 PUBLIC API URL: {public_url}\\n")
    uvicorn.run(app, host="0.0.0.0", port=7860)
'''

with open("app_api.py", "w", encoding="utf-8") as f:
    f.write(code_content)

print("✅ app_api.py အသစ် အဆင်သင့်ဖြစ်ပါပြီ။")