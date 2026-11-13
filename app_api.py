code_content = '''import os
import torch
import soundfile as sf
import numpy as np
import shutil
import re
from datetime import datetime
import gradio as gr
from voxcpm import VoxCPM

print("🔄 Loading VoxCPM model...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
print("✅ Model loaded successfully!")

OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("/tmp", exist_ok=True)

def run_long_tts(text_input, audio_path, prompt_text):
    if not text_input or not text_input.strip():
        raise gr.Error("ကျေးဇူးပြု၍ ထုတ်ယူလိုသော စာသားကို ထည့်ပါ။")
    if not audio_path:
        raise gr.Error("ကျေးဇူးပြု၍ Reference အသံဖိုင်ကို တင်ပါ။")
    if not prompt_text or not prompt_text.strip():
        raise gr.Error("ကျေးဇူးပြု၍ Reference Text ကို ထည့်ပါ။")

    try:
        torch.cuda.empty_cache()
        
        sentences = re.split(r'([။\\.?!\\n]+)', text_input.strip())
        target_chunks = [s.strip() for s in sentences if s.strip()]
        if not target_chunks:
            target_chunks = [text_input.strip()]

        all_wavs = []
        sample_rate = getattr(getattr(model, "tts_model", None), "sample_rate", 48000)
        silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)

        print(f"🔄 စာသားများကို အပိုင်း {len(target_chunks)} ပိုင်းခွဲ၍ အသံထုတ်နေပါပြီ...")

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

        if not all_wavs:
            raise gr.Error("အသံထုတ်လုပ်ရန် အချက်အလက် မရှိပါ။")

        final_wav_np = np.concatenate(all_wavs)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"long_clone_{ts}.wav"
        
        gradio_temp_path = os.path.join("/tmp", filename)
        sf.write(gradio_temp_path, final_wav_np, sample_rate)

        output_path = os.path.join(OUTPUT_DIR, filename)
        shutil.copy(gradio_temp_path, output_path)
        print(f"📁 အသံဖိုင်ကို သိမ်းပြီးပါပြီ: {output_path}")

        return gradio_temp_path

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise gr.Error(f"အမှားအယွင်း ဖြစ်ပေါ်သည်: {str(e)}")

# Gradio Interface (api_name ကို ရှင်းလင်းစွာ သတ်မှတ်ပေးခြင်း)
with gr.Blocks(title="VoxCPM Long-Form Voice Cloning") as demo:
    gr.Markdown("# 🎙️ VoxCPM Long-Form Voice Cloning API")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="စာသားအရှည် ထည့်ရန်", lines=10)
            audio_input = gr.Audio(label="Reference အသံဖိုင် (WAV)", type="filepath")
            prompt_text_input = gr.Textbox(label="Reference Text (အသံဖိုင်ထဲပါသော စာသား)")
            submit_btn = gr.Button("🚀 အသံဖိုင် ထုတ်လုပ်မည်", variant="primary")
        with gr.Column():
            audio_output = gr.Audio(label="ထွက်လာသော အသံဖိုင်", interactive=False)

    # ဤနေရာတွင် api_name="run_long_tts" ကို တိကျစွာ ချိတ်ဆက်ပေးထားပါသည်
    submit_btn.click(
        fn=run_long_tts, 
        inputs=[text_input, audio_input, prompt_text_input], 
        outputs=[audio_output],
        api_name="run_long_tts"
    )

if __name__ == "__main__":
    demo.queue(max_size=10, default_concurrency_limit=2).launch(share=True, debug=True)
'''

with open("app_api.py", "w", encoding="utf-8") as f:
    f.write(code_content)

print("✅ app_api.py ကို အောင်မြင်စွာ ပြင်ဆင်ပြီးပါပြီ။")