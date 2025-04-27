import logging
import os
import re
import subprocess
import yaml # For loading config
import shutil # Needed for moving the vocal file

import faster_whisper
import torch
import torchaudio

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    LANGUAGES,
    TO_LANGUAGE_CODE,
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg, # Keep this helper
    punct_model_langs,
    whisper_langs,
    write_srt,
)

mtypes = {"cpu": "int8", "cuda": "float16"}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}. Please create it or specify the correct path.")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}")
        exit(1)


def separate_vocals(audio_path, temp_dir, output_dir, device, demucs_model):
    """Separates vocals using Demucs, saves final vocal file to output_dir."""
    logging.info(f"Performing source separation on: {audio_path}")
    # Demucs output will still go into temp_dir initially
    # demucs_output_base = os.path.join(temp_dir, demucs_model) # Base within temp dir
    # os.makedirs(demucs_output_base, exist_ok=True)

    # Define Demucs command to output into temp_dir
    demucs_cmd = (
        f'python -m demucs.separate -n {demucs_model} --two-stems=vocals "{audio_path}" '
        f'-o "{temp_dir}" --device "{device}"' # Output to temp_dir first
    )
    logging.info(f"Running Demucs command: {demucs_cmd}")
    return_code = os.system(demucs_cmd)

    # Path where demucs saves the vocal file within temp_dir
    original_basename = os.path.splitext(os.path.basename(audio_path))[0]
    temp_vocal_file_path = os.path.join(temp_dir, demucs_model, original_basename, "vocals.wav")
    logging.info(f"Checking for separated vocals at: {temp_vocal_file_path}")

    if return_code != 0 or not os.path.exists(temp_vocal_file_path):
        # Log as error and raise an exception to halt execution
        error_msg = f"Source splitting failed (exit code: {return_code}) or vocal file not found: {temp_vocal_file_path}. Cannot proceed."
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    else:
        # Define the final destination path in the output directory
        final_vocal_filename = f"{original_basename}_vocals.wav"
        final_vocal_path = os.path.join(output_dir, final_vocal_filename)
        logging.info(f"Moving separated vocals from {temp_vocal_file_path} to {final_vocal_path}")
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            # Move the file
            shutil.move(temp_vocal_file_path, final_vocal_path)
            logging.info(f"Successfully moved vocals to: {final_vocal_path}")
            return final_vocal_path # Return the final path
        except Exception as e:
            error_msg = f"Failed to move vocal file {temp_vocal_file_path} to {final_vocal_path}: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)


def transcribe(audio_path, config):
    """Transcribes audio using Faster Whisper."""
    model_name = config['whisper_model']
    device = config['device']
    language = config.get('language') # Use .get for optional keys
    batch_size = config.get('batch_size', 8) # Default to 8 if not specified
    suppress_numerals = config.get('suppress_numerals', False)
    model_cache_dir = config.get('model_cache_directory') # Optional cache dir

    logging.info(f"Loading Whisper model: {model_name}")
    model = faster_whisper.WhisperModel(
        model_name, 
        device=device, 
        compute_type=mtypes[device], 
        download_root=model_cache_dir # Pass None if not specified
    )
    logging.info(f"Decoding audio file: {audio_path}")
    waveform = faster_whisper.decode_audio(audio_path)

    suppress_tokens = (
        find_numeral_symbol_tokens(model.hf_tokenizer) if suppress_numerals else [-1]
    )

    logging.info("Starting transcription...")
    if batch_size > 0:
        logging.info(f"Using batched inference with batch size: {batch_size}")
        # Use BatchedInferencePipeline for batching
        whisper_pipeline = faster_whisper.BatchedInferencePipeline(model)
        segments, info = whisper_pipeline.transcribe(
            waveform,
            language=language,
            suppress_tokens=suppress_tokens,
            batch_size=batch_size, # Pass batch_size to the pipeline's transcribe
        )
        # Clear pipeline object after use
        del whisper_pipeline
    else:
        logging.info("Using non-batched inference (VAD enabled).")
        # Use model.transcribe directly for non-batched with VAD
        segments, info = model.transcribe(
            waveform,
            language=language,
            suppress_tokens=suppress_tokens,
            vad_filter=True, # batch_size=0 implies using VAD
        )

    full_transcript = "".join(segment.text for segment in segments)
    logging.info(f"Transcription complete. Language detected: {info.language}")

    del model
    torch.cuda.empty_cache()
    return full_transcript, info, waveform # Return waveform for alignment


def align_transcript(audio_waveform, transcript_text, language_code, config):
    """Aligns transcript to audio using CTC Forced Aligner."""
    device = config['device']
    batch_size = config['batch_size']

    logging.info("Loading alignment model...")
    align_model, align_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    logging.info("Generating emissions for alignment...")
    emissions, stride = generate_emissions(
        align_model,
        torch.from_numpy(audio_waveform)
        .to(align_model.dtype)
        .to(align_model.device),
        batch_size=batch_size,
    )

    del align_model
    torch.cuda.empty_cache()

    logging.info("Preprocessing text for alignment...")
    tokens_starred, text_starred = preprocess_text(
        transcript_text,
        romanize=True,
        language=language_code,
    )

    logging.info("Performing alignment...")
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        align_tokenizer,
    )

    logging.info("Getting spans...")
    spans = get_spans(tokens_starred, segments, blank_token)

    logging.info("Postprocessing alignment results...")
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    logging.info("Alignment complete.")
    return word_timestamps


def prepare_nemo_input(vocal_audio_path, temp_dir, nemo_domain):
    """Converts audio to mono WAV for NeMo and creates manifest based on domain."""
    mono_audio_path = os.path.join(temp_dir, "mono_file.wav")
    logging.info(f"Converting to mono for NeMo: {mono_audio_path}")
    waveform, sample_rate = torchaudio.load(vocal_audio_path)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample and save
    torchaudio.save(
        mono_audio_path,
        waveform.float(),
        16000, # NeMo expects 16kHz
        channels_first=True,
    )
    # Create NeMo manifest file using the specified domain
    logging.info(f"Creating NeMo config for domain '{nemo_domain}' pointing to: {temp_dir}")
    config = create_config(temp_dir, nemo_domain) # Pass domain to helper
    return config, mono_audio_path

def run_nemo_diarization(config, device):
    """Runs NeMo MSDD model for diarization."""
    logging.info("Initializing NeMo MSDD model...")
    msdd_model = NeuralDiarizer(cfg=config).to(device)
    logging.info("Starting NeMo diarization...")
    msdd_model.diarize()
    logging.info("NeMo diarization complete.")
    del msdd_model
    torch.cuda.empty_cache()


def get_speaker_timestamps(rttm_file_path):
    """Reads speaker timestamps from RTTM file."""
    logging.info(f"Reading speaker timestamps from RTTM: {rttm_file_path}")
    speaker_ts = []
    try:
        with open(rttm_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000) # Start time in ms
                e = s + int(float(line_list[8]) * 1000) # End time in ms
                speaker_label = int(line_list[11].split("_")[-1]) # Speaker ID
                speaker_ts.append([s, e, speaker_label])
    except FileNotFoundError:
        logging.error(f"RTTM file not found: {rttm_file_path}")
        return [] # Return empty list if RTTM is missing
    except Exception as e:
        logging.error(f"Error reading RTTM file {rttm_file_path}: {e}")
        return []
    return speaker_ts


def process_transcript_and_mapping(word_timestamps, speaker_timestamps, language, config):
    """Maps words to speakers, restores punctuation, and creates sentence mappings."""
    punctuation_model_name = config['punctuation_model']

    logging.info("Mapping words to speakers...")
    wsm = get_words_speaker_mapping(word_timestamps, speaker_timestamps, "start")

    if language in punct_model_langs:
        logging.info(f"Restoring punctuation for language: {language} using {punctuation_model_name}")
        punct_model = PunctuationModel(model=punctuation_model_name)
        words_list = list(map(lambda x: x["word"], wsm))
        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            # Check if word is not None or empty string
            if word and \
               labeled_tuple[1] in ending_puncts and \
               (not word.endswith(tuple(model_puncts)) or is_acronym(word)):
                word += labeled_tuple[1]
                if word.endswith(".."): word = word.rstrip(".") # Avoid double periods
                word_dict["word"] = word
    else:
        logging.warning(
            f"Punctuation restoration not available for {language}. Using original punctuation."
        )

    logging.info("Realigning word-speaker mapping with punctuation...")
    wsm_realigned = get_realigned_ws_mapping_with_punctuation(wsm)
    logging.info("Generating sentence-speaker mapping...")
    ssm = get_sentences_speaker_mapping(wsm_realigned, speaker_timestamps)
    logging.info("Sentence mapping complete.")
    return ssm

def save_outputs(ssm, audio_file_path, output_dir):
    """Saves the transcript in TXT and SRT formats into the specified output directory."""
    if not output_dir:
        logging.warning("Output directory not specified. Skipping saving of main TXT/SRT files.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Use the basename of the original audio for the output filenames
    output_basename = os.path.splitext(os.path.basename(audio_file_path))[0]
    txt_path = os.path.join(output_dir, f"{output_basename}.txt")
    srt_path = os.path.join(output_dir, f"{output_basename}.srt")

    logging.info(f"Saving main transcript to: {txt_path}")
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    logging.info(f"Saving SRT to: {srt_path}")
    with open(srt_path, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    logging.info("Output files saved.")

def generate_tts_dataset(ssm, vocal_audio_path, output_dir, original_audio_basename, config):
    """Generates the TTS dataset with speaker folders, audio chunks, and transcripts."""
    if not output_dir:
        logging.warning("output_tts_directory not specified in config. Skipping TTS dataset generation.")
        return

    # Get minimum duration from config, default to 0 (no filtering) if not specified
    min_duration_ms = config.get('minimum_chunk_duration_ms', 0)
    logging.info(f"Minimum chunk duration for TTS dataset set to: {min_duration_ms}ms")

    logging.info(f"Generating TTS dataset in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load the full vocal audio once
    try:
        logging.info(f"Loading full vocal audio for chunking: {vocal_audio_path}")
        vocal_waveform, vocal_sample_rate = torchaudio.load(vocal_audio_path)
    except Exception as e:
        logging.error(f"Failed to load vocal audio {vocal_audio_path}: {e}. Cannot generate TTS dataset.")
        return

    for i, segment in enumerate(ssm):
        # Extract speaker ID, ensuring it's a valid directory name
        speaker_label_raw = segment.get('speaker', 'Unknown_Speaker')
        speaker_label = re.sub(r'[\s/:]+', '_', speaker_label_raw) # Replace spaces/slashes etc.
        text = segment.get('text', '').strip()
        start_ms = segment.get('start_time')
        end_ms = segment.get('end_time')

        if not text or start_ms is None or end_ms is None:
            logging.warning(f"Skipping segment {i} for {speaker_label} due to missing text or timestamps.")
            continue

        # Calculate duration and check against minimum
        duration_ms = end_ms - start_ms
        if duration_ms < min_duration_ms:
            logging.info(f"Skipping segment {i} for {speaker_label} due to short duration ({duration_ms}ms < {min_duration_ms}ms). Text: \"{text}\"")
            continue

        # Create speaker-specific directories
        speaker_audio_dir = os.path.join(output_dir, speaker_label, "audio")
        speaker_text_dir = os.path.join(output_dir, speaker_label, "text")
        try:
            os.makedirs(speaker_audio_dir, exist_ok=True)
            os.makedirs(speaker_text_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create directories for {speaker_label}: {e}")
            continue # Skip this segment if dirs can't be created

        # Calculate start and end frames
        start_frame = int(start_ms / 1000 * vocal_sample_rate)
        end_frame = int(end_ms / 1000 * vocal_sample_rate)
        end_frame = min(end_frame, vocal_waveform.shape[1]) # Ensure within bounds
        start_frame = max(0, start_frame)

        # Slice the waveform
        if start_frame >= end_frame:
            logging.warning(f"Skipping segment {i} for {speaker_label} due to invalid time range (start={start_ms}ms, end={end_ms}ms). Text: \"{text}\"")
            continue

        audio_chunk = vocal_waveform[:, start_frame:end_frame]

        if audio_chunk.numel() == 0:
            logging.warning(f"Skipping segment {i} for {speaker_label} due to empty audio chunk (start={start_ms}ms, end={end_ms}ms). Text: \"{text}\"")
            continue

        # Define output filenames (ensure uniqueness)
        chunk_basename = f"{original_audio_basename}_spk_{speaker_label}_chunk_{i:04d}"
        output_wav_path = os.path.join(speaker_audio_dir, f"{chunk_basename}.wav")
        output_txt_path = os.path.join(speaker_text_dir, f"{chunk_basename}.txt")

        # Save audio chunk
        try:
            torchaudio.save(output_wav_path, audio_chunk, vocal_sample_rate)
        except Exception as e:
            logging.error(f"Failed to save audio chunk {output_wav_path}: {e}")
            continue # Skip saving text if audio failed

        # Save transcription text (only the text)
        try:
            with open(output_txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)
        except Exception as e:
            logging.error(f"Failed to save text file {output_txt_path}: {e}")
            # Clean up the corresponding audio file if text saving fails
            if os.path.exists(output_wav_path):
                try:
                    os.remove(output_wav_path)
                    logging.info(f"Removed corresponding audio chunk {output_wav_path} due to text saving error.")
                except OSError as rm_e:
                    logging.error(f"Failed to remove audio chunk {output_wav_path} after text error: {rm_e}")

    logging.info("TTS dataset generation complete.")

def main():
    # Load configuration directly, assuming config.yaml in the same directory
    config = load_config()

    # Get audio path from config
    audio_path = config.get('input_audio')
    if not audio_path or not os.path.exists(audio_path):
        logging.error(f"Input audio file specified in config ('{audio_path}') not found or not specified.")
        return # Exit if audio path is invalid

    # Determine device (handle 'auto')
    if config['device'].lower() == 'auto':
        config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {config['device']}")

    # Process language (handle null)
    language = process_language_arg(config.get('language'), config['whisper_model'])

    # Define output and temporary directories from config
    output_tts_dir = config.get('output_tts_directory')
    if not output_tts_dir:
        logging.warning("output_tts_directory not specified in config. Vocal file might not be saved persistently.")
        # Decide fallback behaviour - maybe save next to input audio? For now, log warning.
        # output_tts_dir = os.path.dirname(audio_path) # Example fallback

    temp_dir_base = config.get('temp_directory_base', 'temp_outputs') # Default if missing
    temp_dir = os.path.join(temp_dir_base, os.path.splitext(os.path.basename(audio_path))[0])
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 1. Vocal Separation - Pass output_tts_dir
        vocal_target_path = separate_vocals(
            audio_path, 
            temp_dir,
            output_tts_dir, # Pass the target directory for the final vocal file
            config['device'],
            config['demucs_model']
        )

        # 2. Transcription
        full_transcript, lang_info, audio_waveform = transcribe(
            vocal_target_path,
            config # Pass the whole config dict
        )
        detected_language = lang_info.language
        if detected_language not in langs_to_iso:
             logging.error(f"Detected language '{detected_language}' not supported for alignment.")
             return # Exit if language is not alignable
        language_iso = langs_to_iso[detected_language]

        # 3. Forced Alignment
        word_timestamps = align_transcript(
            audio_waveform,
            full_transcript,
            language_iso,
            config # Pass the whole config dict
        )

        # Clear waveform from memory if no longer needed
        del audio_waveform
        torch.cuda.empty_cache()

        # 4. Prepare NeMo Input
        nemo_config_obj, mono_audio_path = prepare_nemo_input(
            vocal_target_path,
            temp_dir,
            config['nemo_domain']
        )

        # 5. Run NeMo Diarization
        run_nemo_diarization(nemo_config_obj, config['device'])

        # 6. Get Speaker Timestamps from RTTM
        rttm_file = os.path.join(temp_dir, "pred_rttms", "mono_file.rttm")
        speaker_timestamps = get_speaker_timestamps(rttm_file)

        if not speaker_timestamps:
            logging.error("Speaker diarization failed or produced no timestamps. Cannot proceed.")
            return

        # 7. Process Transcript and Mapping
        ssm = process_transcript_and_mapping(
            word_timestamps,
            speaker_timestamps,
            detected_language, # Use detected language for punctuation
            config # Pass the whole config dict
        )

        # 8. Save Outputs (Original TXT/SRT) - Pass output_tts_dir
        save_outputs(ssm, audio_path, output_tts_dir)

        # 9. Generate TTS Dataset
        generate_tts_dataset(
            ssm,
            vocal_target_path, # Use the path to the (potentially separated) vocals
            config.get('output_tts_directory'), # Get dir from config
            os.path.splitext(os.path.basename(audio_path))[0], # Original basename for chunk naming
            config # Pass the whole config dict
        )

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True)
    finally:
        # 10. Cleanup
        logging.info(f"Cleaning up temporary directory: {temp_dir}")
        try:
            if os.path.exists(temp_dir):
                 cleanup(temp_dir)
        except Exception as e:
             logging.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")

    logging.info("Processing finished.")


if __name__ == "__main__":
    main()
