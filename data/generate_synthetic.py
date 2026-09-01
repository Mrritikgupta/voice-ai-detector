import asyncio
import time
import edge_tts
from gtts import gTTS
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config_v2 as config

config.ensure_dirs()

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Technology is changing the way we live and work every single day.",
    "Please describe a challenging project you worked on recently.",
    "I believe strong communication skills are essential for teamwork.",
    "Can you walk me through your experience with data analysis.",
    "The weather today is quite pleasant for a walk in the park.",
    "Our company values innovation, integrity, and collaboration.",
    "What motivated you to apply for this position.",
    "Machine learning models require large amounts of quality data.",
    "Tell me about a time you resolved a conflict at work.",
    "The interview process typically involves multiple rounds of screening.",
    "I am passionate about solving real world problems with technology.",
    "How do you handle tight deadlines and pressure at work.",
    "Effective leadership requires empathy and clear decision making.",
    "This role requires strong analytical and problem solving skills.",
    "Describe your greatest professional achievement to date.",
    "I have five years of experience working in software development.",
    "What are your salary expectations for this position.",
    "Continuous learning is important in a fast changing industry.",
    "Thank you for taking the time to interview me today.",
    "Can you tell me a little about yourself and your background.",
    "Why do you want to leave your current job.",
    "What are your greatest strengths and weaknesses.",
    "Where do you see yourself in the next five years.",
    "Describe a situation where you had to work under pressure.",
    "How do you prioritize tasks when managing multiple projects.",
    "What do you know about our company and its products.",
    "Tell me about a time you failed and what you learned.",
    "How would your previous manager describe your work ethic.",
    "What kind of work environment helps you perform your best.",
    "Do you have any questions for us about the role.",
    "Explain a complex technical concept in simple terms.",
    "How do you stay updated with the latest industry trends.",
    "Describe your approach to learning a new skill quickly.",
    "What tools and technologies are you most comfortable using.",
    "How do you handle feedback and criticism from colleagues.",
    "Tell me about a successful team project you contributed to.",
    "What steps do you take to ensure quality in your work.",
    "How do you balance speed and accuracy in your tasks.",
    "Describe a time you had to learn something completely new.",
    "What makes you a good fit for this particular role.",
    "How do you approach solving a problem you have never seen before.",
    "Tell me about your experience working with remote teams.",
    "What is your process for debugging a difficult issue.",
    "How do you handle disagreements with your teammates.",
    "Describe your ideal work culture and management style.",
    "What accomplishments are you most proud of in your career.",
    "How do you keep yourself motivated during long projects.",
    "Tell me about a time you had to persuade someone.",
    "What is the most valuable lesson you have learned professionally.",
]

EDGE_VOICES = [
    "en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural",
    "en-GB-SoniaNeural", "en-IN-PrabhatNeural", "en-IN-NeerjaNeural",
    "en-AU-WilliamNeural", "en-AU-NatashaNeural",
]

REQUEST_DELAY_SEC = 0.5


async def generate_edge_tts(text, voice, out_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def convert_to_wav(mp3_path, wav_path):
    audio, sr = librosa.load(mp3_path, sr=config.SAMPLE_RATE, mono=True)
    sf.write(wav_path, audio, config.SAMPLE_RATE)
    return len(audio) / config.SAMPLE_RATE


def generate_edge_batch(rows, counter):
    print("Generating edge-tts samples...")
    total = len(SENTENCES) * len(EDGE_VOICES)
    done = 0
    for sentence in SENTENCES:
        for voice in EDGE_VOICES:
            counter["n"] += 1
            sample_id = f"synth_edge_{counter['n']:04d}"
            mp3_path = config.RAW_DOWNLOADS_DIR / f"{sample_id}.mp3"
            wav_path = config.RAW_DIGITAL_DIR / f"{sample_id}.wav"

            try:
                asyncio.run(generate_edge_tts(sentence, voice, str(mp3_path)))
                duration = convert_to_wav(mp3_path, wav_path)
                mp3_path.unlink(missing_ok=True)

                rows.append({
                    "sample_id": sample_id,
                    "filepath": str(wav_path),
                    "source_filepath": "",
                    "parent_sample_id": "",
                    "transform_id": "",
                    "label": config.LABEL_DIGITAL_SYNTHETIC,
                    "delivery_mode": config.DELIVERY_DIGITAL_INJECTION,
                    "content_type": config.CONTENT_TTS,
                    "generation_type": "SYNTHETIC",
                    "generator_id": "edge_tts",
                    "speaker_id": voice,
                    "dataset": "synthetic_unseen_edge_tts",
                    "attack_id": "edge_tts",
                    "replay": False,
                    "replay_type": "",
                    "codec": "",
                    "device": "cloud_tts",
                    "language": "en",
                    "accent": voice.split("-")[1] if "-" in voice else "",
                    "rir_id": "",
                    "sample_rate": config.SAMPLE_RATE,
                    "duration_sec": round(duration, 3),
                    "recording_session_id": "",
                    "room_id": "",
                    "channel_condition_id": "",
                    "license_source": "edge_tts",
                    "is_augmented": False,
                    "split": "unseen_generator_test",
                })
                done += 1
                if done % 20 == 0:
                    print(f"  [{done}/{total}] edge-tts samples done.")

            except Exception as e:
                print(f"  Skipping edge-tts sample ({voice}): {e}")

            time.sleep(REQUEST_DELAY_SEC)


def generate_gtts_batch(rows, counter):
    print("Generating gTTS samples...")
    for i, sentence in enumerate(SENTENCES):
        counter["n"] += 1
        sample_id = f"synth_gtts_{counter['n']:04d}"
        mp3_path = config.RAW_DOWNLOADS_DIR / f"{sample_id}.mp3"
        wav_path = config.RAW_DIGITAL_DIR / f"{sample_id}.wav"

        try:
            tts = gTTS(text=sentence, lang="en")
            tts.save(str(mp3_path))
            duration = convert_to_wav(mp3_path, wav_path)
            mp3_path.unlink(missing_ok=True)

            rows.append({
                "sample_id": sample_id,
                "filepath": str(wav_path),
                "source_filepath": "",
                "parent_sample_id": "",
                "transform_id": "",
                "label": config.LABEL_DIGITAL_SYNTHETIC,
                "delivery_mode": config.DELIVERY_DIGITAL_INJECTION,
                "content_type": config.CONTENT_TTS,
                "generation_type": "SYNTHETIC",
                "generator_id": "gtts",
                "speaker_id": "gtts_default",
                "dataset": "synthetic_unseen_gtts",
                "attack_id": "gtts",
                "replay": False,
                "replay_type": "",
                "codec": "",
                "device": "cloud_tts",
                "language": "en",
                "accent": "",
                "rir_id": "",
                "sample_rate": config.SAMPLE_RATE,
                "duration_sec": round(duration, 3),
                "recording_session_id": "",
                "room_id": "",
                "channel_condition_id": "",
                "license_source": "gtts",
                "is_augmented": False,
                "split": "unseen_generator_test",
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(SENTENCES)}] gTTS samples done.")
        except Exception as e:
            print(f"  Skipping gTTS sample {i}: {e}")

        time.sleep(REQUEST_DELAY_SEC)


def main():
    rows = []
    counter = {"n": 0}

    generate_edge_batch(rows, counter)
    generate_gtts_batch(rows, counter)

    new_df = pd.DataFrame(rows, columns=config.MANIFEST_COLUMNS)

    if config.MASTER_MANIFEST_PATH.exists():
        existing_df = pd.read_csv(config.MASTER_MANIFEST_PATH, low_memory=False)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(config.MASTER_MANIFEST_PATH, index=False)

    print(f"\nDone. {len(new_df)} new synthetic samples added to manifest.")
    print(f"Total manifest size: {len(combined_df)} rows")
    print(new_df["dataset"].value_counts())


if __name__ == "__main__":
    main()