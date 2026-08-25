import gradio as gr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import detector
import audio_utils
import config


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-void: #0A0E14;
    --bg-panel: #10151D;
    --bg-panel-light: #171E28;
    --border-soft: #232B37;
    --teal: #00E5C7;
    --teal-dim: #00E5C733;
    --green: #34D399;
    --green-dim: #34D39933;
    --red: #FF4D4D;
    --red-dim: #FF4D4D33;
    --text-hi: #E6EDF3;
    --text-mid: #8B98A5;
    --text-low: #4B5563;
}

.gradio-container {
    background: radial-gradient(ellipse at top, #0D1420 0%, #0A0E14 60%) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-hi) !important;
}

#header {
    text-align: center;
    padding: 28px 0 8px 0;
}

#header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    letter-spacing: 0.02em;
    margin: 0;
    background: linear-gradient(90deg, #E6EDF3 0%, #00E5C7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#header .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: var(--teal);
    text-transform: uppercase;
    margin-bottom: 6px;
}

#header .subtitle {
    color: var(--text-mid);
    font-size: 0.92rem;
    margin-top: 6px;
}

.panel {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 14px !important;
}

#analyze-btn {
    background: linear-gradient(135deg, #00E5C7 0%, #00B8A0 100%) !important;
    color: #06110E !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 0 24px var(--teal-dim) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

#analyze-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 32px var(--teal-dim) !important;
}

#verdict-box {
    min-height: 260px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.verdict-idle {
    text-align: center;
    color: var(--text-low);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.verdict-card {
    width: 100%;
    text-align: center;
    padding: 20px 0;
    animation: reveal 0.4s ease;
}

@keyframes reveal {
    from { opacity: 0; transform: scale(0.96); }
    to { opacity: 1; transform: scale(1); }
}

.scan-ring {
    width: 120px;
    height: 120px;
    margin: 0 auto 18px auto;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    position: relative;
}

.ring-real {
    border: 3px solid var(--green);
    box-shadow: 0 0 30px var(--green-dim), inset 0 0 20px var(--green-dim);
    color: var(--green);
}

.ring-fake {
    border: 3px solid var(--red);
    box-shadow: 0 0 30px var(--red-dim), inset 0 0 20px var(--red-dim);
    color: var(--red);
    animation: pulse-alert 1.1s infinite;
}

@keyframes pulse-alert {
    0%, 100% { box-shadow: 0 0 30px var(--red-dim), inset 0 0 20px var(--red-dim); }
    50% { box-shadow: 0 0 46px var(--red-dim), inset 0 0 28px var(--red-dim); }
}

.verdict-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

.label-real { color: var(--green); }
.label-fake { color: var(--red); }

.verdict-sub {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-mid);
    font-size: 0.78rem;
    letter-spacing: 0.05em;
}

.meter-track {
    width: 80%;
    height: 6px;
    background: var(--bg-panel-light);
    border-radius: 4px;
    margin: 16px auto 0 auto;
    overflow: hidden;
    border: 1px solid var(--border-soft);
}

.meter-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

#details-box {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-mid) !important;
    background: var(--bg-panel-light) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 10px !important;
}

footer { display: none !important; }
"""


def build_verdict_html(label, prob, num_chunks, threshold):
    is_fake = label == "AI-GENERATED"
    ring_class = "ring-fake" if is_fake else "ring-real"
    label_class = "label-fake" if is_fake else "label-real"
    display_pct = prob * 100 if is_fake else (1 - prob) * 100
    verdict_text = "AI-GENERATED" if is_fake else "REAL HUMAN"
    meter_color = "var(--red)" if is_fake else "var(--green)"

    return f"""
    <div class="verdict-card">
        <div class="scan-ring {ring_class}">{display_pct:.0f}%</div>
        <div class="verdict-label {label_class}">{verdict_text}</div>
        <div class="verdict-sub">CONFIDENCE {display_pct:.1f}%  ·  {num_chunks} SEGMENT(S) ANALYZED</div>
        <div class="meter-track">
            <div class="meter-fill" style="width:{display_pct:.1f}%; background:{meter_color};"></div>
        </div>
    </div>
    """


IDLE_HTML = """
<div class="verdict-idle">
    ◈ &nbsp; AWAITING AUDIO INPUT &nbsp; ◈<br>
    <span style="opacity:0.6; font-size:0.75rem;">record or upload, then run scan</span>
</div>
"""


def analyze(audio_path):
    if audio_path is None:
        return IDLE_HTML, "No audio provided yet."

    result = detector.check_audio_file(audio_path)

    verdict_html = build_verdict_html(
        result["label"], result["probability_fake"],
        result["num_chunks"], detector.DECISION_THRESHOLD
    )

    details = (
        f"P(fake)        : {result['probability_fake']*100:.2f}%\n"
        f"Threshold      : {detector.DECISION_THRESHOLD}\n"
        f"Chunks         : {result['num_chunks']}\n"
        f"Sample rate    : {config.SAMPLE_RATE} Hz\n"
        f"Model          : WavLM-base + classifier head"
    )

    return verdict_html, details


with gr.Blocks(title="Voice Authenticity Scanner", css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
    gr.HTML("""
        <div id="header">
            <div class="eyebrow">SHERLOCK AI &nbsp;·&nbsp; VOICE FORENSICS</div>
            <h1>Voice Authenticity Scanner</h1>
            <div class="subtitle">Detects AI-cloned and synthetic speech in live audio segments</div>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["panel"]):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="AUDIO INPUT"
            )
            analyze_btn = gr.Button("▶  RUN SCAN", elem_id="analyze-btn", size="lg")

        with gr.Column(scale=1, elem_classes=["panel"]):
            result_html = gr.HTML(IDLE_HTML, elem_id="verdict-box")

    details_box = gr.Textbox(
        label="SCAN DETAILS",
        interactive=False,
        elem_id="details-box",
        lines=5
    )

    analyze_btn.click(
        fn=analyze,
        inputs=audio_input,
        outputs=[result_html, details_box]
    )

if __name__ == "__main__":
    print("Warming up model...")
    detector.load_detector()
    print("Ready.")
    demo.launch(share=True)