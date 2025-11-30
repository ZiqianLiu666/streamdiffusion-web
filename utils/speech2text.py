import os
import tempfile
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline


class SpeechToText:
    def __init__(self,
                 model_id: str = "openai/whisper-small",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Use HuggingFace Whisper for speech recognition, uniformly use FP32 (most stable).
        """
        self.device_str = device
        self.device_index = 0 if device == "cuda" and torch.cuda.is_available() else -1

        print(f"🎙️ Loading HF Whisper model ({model_id}) on {self.device_str} ...")

        # Whisper input is FP32, so model must also be FP32
        torch_dtype = torch.float32

        # Load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(self.device_str)

        # Processor (contains tokenizer + feature extractor)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Whisper pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=self.device_index,
        )

        print("✅ HF Whisper model ready (FP32).")

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        # Write audio bytes to wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = self.pipe(
                tmp_path,
                generate_kwargs={
                    "language": language,
                    "task": "transcribe",
                },
            )
            text = result["text"].strip()
            print(f"🗣️ Recognized text: {text}")
            return text
        finally:
            os.remove(tmp_path)
