# TTS Dataset Generator using Whisper Diarization

This is a fork of **[whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)** 

## Purpose 

The primary goal of this fork is to adapt the original speaker diarization pipeline to **streamline the creation of Text-To-Speech (TTS) datasets**.

While the original tool provides timed speaker transcripts (`.srt`), this fork modifies the process to **directly output speaker-specific audio chunks and their corresponding transcription text files**, reducing the manual steps needed for TTS data preparation.

It still leverages the core strengths of the original pipeline:

1.  Accurate transcription via Whisper.
2.  Speaker segment identification via NeMo diarization.
3.  Precise timing information.

## Key Changes in this Fork

*   **Configuration File:** Uses a `config.yaml` file to specify input audio and output paths, instead of relying solely on command-line arguments.
*   **Automated Output:** Directly generates segmented audio chunks (`.wav`) and individual transcription files (`.txt`) organized by speaker, in addition to the standard `.srt` file.

## Installation

This fork requires the same dependencies as the original `whisper-diarization`. Please follow the installation instructions provided in the **[original repository's README](https://github.com/MahmoudAshraf97/whisper-diarization#️-installation)**.

*(No additional dependencies specific to this fork's primary purpose have been added unless your specific code changes require them).*

## Usage for TTS Dataset Preparation

1.  **Configure `config.yaml`:** Create or edit the `config.yaml` file in the repository root. Specify at least the following:
    ```yaml
    # Example config.yaml structure (adjust based on your actual implementation)
    audio_path: "/path/to/your/input/audio.mp3"
    output_directory: "/path/to/your/output/dataset_folder"
    ```
2.  **Run the Script:** Execute the main diarization script from the repository root:
    ```bash
    python diarize.py
    ```
3.  **Check the Output:** Navigate to the `output_directory` you specified in `config.yaml`. You should find:
    *   An `.srt` file containing the full diarized transcript (e.g., `audio.srt`).
    *   Subdirectories for each identified speaker (e.g., `speaker_0/`, `speaker_1/`, etc.).
    *   Inside each speaker directory:
        *   Numbered audio chunk files (e.g., `segment_001.wav`, `segment_002.wav`, ...).
        *   Corresponding transcription text files (e.g., `segment_001.txt`, `segment_002.txt`, ...).
