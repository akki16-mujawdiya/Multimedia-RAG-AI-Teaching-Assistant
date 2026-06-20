# Multimedia RAG-Based AI Teaching Assistant

## Overview

The Multimedia RAG-Based AI Teaching Assistant is an intelligent educational system that allows users to create a personalized AI tutor from their own video content.

The system processes educational videos, converts speech into text, generates structured knowledge representations, creates embeddings, and retrieves relevant information to answer user questions using Retrieval-Augmented Generation (RAG).

This project demonstrates the practical implementation of Speech Recognition, Natural Language Processing (NLP), Semantic Search, Embeddings, and Generative AI technologies.

---

## How It Works

### Step 1: Collect Educational Videos

Place all educational video files inside the `videos` folder.

### Step 2: Convert Videos to Audio

Run:

```bash
python video_to_mp3.py
```

This extracts audio from all video files and stores them as MP3 files.

### Step 3: Convert Audio to JSON

Run:

```bash
python mp3_to_json.py
```

The audio is transcribed using Whisper Speech Recognition and stored in JSON format.

### Step 4: Generate Embeddings

Run:

```bash
python preprocess_json.py
```

The JSON files are processed and converted into vector embeddings.

The embeddings are stored in:

```text
embeddings.joblib
```

### Step 5: Retrieval-Augmented Generation (RAG)

The system:

1. Loads vector embeddings
2. Retrieves the most relevant educational content
3. Builds a contextual prompt
4. Sends the prompt to an LLM
5. Generates an intelligent answer

---

## System Architecture

Educational Videos

↓

Video to Audio Conversion

↓

Speech Recognition (Whisper)

↓

JSON Knowledge Extraction

↓

Embedding Generation

↓

Vector Storage

↓

Semantic Search

↓

Prompt Construction

↓

Large Language Model (LLM)

↓

AI Generated Answer

---

## Features

* Video-based Knowledge Processing
* Speech-to-Text Conversion
* Whisper Integration
* JSON Knowledge Extraction
* Embedding Generation
* Semantic Retrieval
* Retrieval-Augmented Generation (RAG)
* AI Teaching Assistant
* Educational Question Answering

---

## Technologies Used

* Python
* OpenAI Whisper
* NLP
* Embeddings
* Joblib
* JSON Processing
* Retrieval-Augmented Generation (RAG)
* Large Language Models (LLMs)

---

## Project Structure

RAG BASED PROJECT 2/

├── videos/

├── audios/

├── jsons/

├── whisper/

├── config.py

├── video_to_mp3.py

├── mp3_to_json.py

├── preprocess_json.py

├── process_incoming.py

├── speechtotext.py

├── embeddings.joblib

├── prompt.txt

├── response.txt

└── README.md

---

## Learning Outcomes

* Retrieval-Augmented Generation (RAG)
* Speech Recognition Systems
* Semantic Search
* Embedding Models
* Knowledge Retrieval
* Educational AI Systems
* End-to-End AI Pipeline Development

---

## Future Enhancements

* Streamlit Web Application
* Multi-Video Knowledge Base
* Voice-Based Question Answering
* Quiz Generation
* Learning Recommendations
* Multi-Language Support

---

## Author

Akki Mujawdiya

Aspiring AI/ML Engineer | Data Science & Generative AI Enthusiast

GitHub: https://github.com/akki16-mujawdiya
