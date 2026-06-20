import whisper
import json
import os

model = whisper.load_model("large-v2")

audios = os.listdir("audios")

for audio in audios:
    print(audio)

    if "_" in audio:

        # Example:
        # 14_144p.mp4_Introduction to CSS Sigma Web Development Course.mp3

        parts = audio.split("_")

        number = parts[0]

        # Skip "144p.mp4" and join the remaining title
        title = "_".join(parts[2:])[:-4]

        print(number, title)

        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )

        chunks = []

        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, indent=4)

print("Done")
