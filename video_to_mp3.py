#converts videos to mp3
# import os
# import subprocess


# files = os.listdir("videos")
# print(files)

# for file in files:
#     print(file)
#     tutorial_number= file.split("[")[0].split("#")[1]
#     file_name= file.split("-")[0]
#     print(tutorial_number, file_name)
#     subprocess.run(["ffmpeg", "-i",f"videos/{file}" , f"audios/{tutorial_number}_{file_name}.mp3"])
    

# Converts videos to mp3 with clean filenames

import os
import subprocess

videos = os.listdir("videos")

for file in videos:

    if not file.endswith((".mp4", ".mkv", ".webm")):
        continue

    print("Processing:", file)

    # Example filename:
    # #1 - Installing VS Code & How Websites Work | Sigma Web Development Course.mp4

    try:
        tutorial_number = file.split("#")[1].split("-")[0].strip()

        title = file.split("-")[1].rsplit(".", 1)[0].strip()

        # Remove characters invalid in Windows filenames
        invalid_chars = r'<>:"/\|?*'
        for ch in invalid_chars:
            title = title.replace(ch, "")

        output_name = f"{tutorial_number}_{title}.mp3"

        subprocess.run([
            "ffmpeg",
            "-i",
            os.path.join("videos", file),
            os.path.join("audios", output_name)
        ])

        print("Created:", output_name)

    except Exception as e:
        print(f"Skipped {file}")
        print(e)