
import whisper
import moviepy.editor as mp
from transformers import pipeline

video_path = "lecture1.mp4"
audio_path = "audio.wav"

video = mp.VideoFileClip(video_path)
video.audio.write_audiofile(audio_path)

model = whisper.load_model("base")
result = model.transcribe(audio_path)

transcript = result["text"]
with open("transcript.txt","w") as f:
    f.write(transcript)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(transcript, max_length=100, min_length=40, do_sample=False)

with open("summary.txt","w") as f:
    f.write(summary[0]["summary_text"])
