import sys, torch, os, argparse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def transcribe_audio(audio_file: str, language: str, model_id: str) -> str:
    print(f"Loading model: {model_id}...")
    
    use_cuda = torch.cuda.is_available()
    torch_device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    pipeline_device = 0 if use_cuda else -1
    
    # Use half precision, faster without compromising much accuracy
    dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    model.to(torch_device)
    print(f"Using device: {torch_device}")

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30, # 30 seconds per chunk
        batch_size=12 if use_cuda else 1,
        return_timestamps=False,
        dtype=dtype,
        device=pipeline_device,
        ignore_warning=True
    )
    print("Transcribing...")
    result = pipe(audio_file, generate_kwargs={"language": language})
    
    return result["text"]


if __name__ == "__main__":
    # argparse stuff
    parser = argparse.ArgumentParser(description="Transcribe mp3 files using Whisper")
    parser.add_argument(
        "audio_file",
        help="Path to the mp3 file to transcribe")
    parser.add_argument(
        "-l", "--language",
        default="english", choices=["english", "swedish"],
        help="Language of the audio (default: english)")
    parser.add_argument(
        "-o", "--output",
        default="transcript.txt",
        help="Output file path (default: transcript.txt)")
    args = parser.parse_args()

    # Model selection
    if args.language == "swedish":
        model_id = "KBLab/kb-whisper-large"
    else:
        model_id = "openai/whisper-large-v3-turbo"

    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found")
        sys.exit(1)

    try:
        transcript = transcribe_audio(args.audio_file, args.language, model_id)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(transcript)
            
        print(f"\nSuccess! Transcription saved to '{args.output}'")
        
    except Exception as e:
        print(f"An error occurred: {e}")