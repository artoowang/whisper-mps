# To run:
# python run_whisper_mps.py --file-name ~ollama/log/wyoming-stt-audio-debug/speech_20251020_091445.wav --model-name large

from whisper_mps import whisper
from whisper_mps.utils.ytdownloader import download_and_convert_to_mp3
from whisper_mps.whisper.transcribe import ModelHolder
import argparse
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
import json
import logging
import mlx.core as mx
import time


parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
parser.add_argument(
    "--file-name",
    required=False,
    type=str,
    help="Path to the audio file to be transcribed.",
)

parser.add_argument(
    "--model-name",
    required=False,
    default="tiny",
    type=str,
    help="Name of the whisper model size. (default: tiny)",
)

parser.add_argument(
    "--youtube-url",
    required=False,
    default=None,
    type=str,
    help="the address from Youtube,like: https://www.youtube.com/watch?v=jaM02mb6JFM",
)

parser.add_argument(
    "--output-file-name",
    required=False,
    default="output.json",
    type=str,
    help="the output file name for the transcribed text JSON",
)

def worker(file_name,model_name,output_file_name):
    start_time = time.time()
    text = whisper.transcribe(file_name,model=model_name)
    logging.info(f'Transcription completed in {time.time() - start_time} seconds.')
    print(text)
    with open(output_file_name, "w", encoding="utf8") as fp:
        json.dump(text, fp, ensure_ascii=False)
    print(
        f"Voila!✨ Your file has been transcribed go check it out over here 👉 {output_file_name}"
    )

def main():
    args = parser.parse_args()
    file_name = args.file_name
    model_name = args.model_name
    youtube_url = args.youtube_url
    output_file_name = args.output_file_name

    # Preloading the model. Note this needs to match the fp16 option passed to
    # transcribe. When not specified, fp16 is True by default.
    logging.info(f'Preloading the model: {model_name} ...')
    ModelHolder.get_model(model_name, mx.float16)
    logging.info('Model loaded.')

    if not output_file_name.lower().endswith('.json'):
        output_file_name = output_file_name + '.json'
    if youtube_url is not None:
        print(f'start downloading audios: {args.youtube_url}')
        audio_path = download_and_convert_to_mp3(youtube_url)
        worker(audio_path,model_name,output_file_name)
    else:
        if file_name is None:
            logging.error(f"local file_name should not be none!")
            return None
        worker(file_name,model_name,output_file_name)    

main()
