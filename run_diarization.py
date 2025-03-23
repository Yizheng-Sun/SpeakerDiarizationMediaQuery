import os
import subprocess
import sys
import numpy as np
import torch
from pyannote.audio import Pipeline
import ffmpeg
import wave
import contextlib
import datetime
import json
import math
import tempfile
import shutil
import time  # Added for timing measurements
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

class TimingStats:
    """Class to track timing information for different processing stages."""
    def __init__(self):
        self.audio_extraction_time = 0
        self.diarization_time = 0
        self.transcription_time = 0
        self.chunk_diarization_times = []
        self.segment_transcription_times = []
        
    def to_dict(self):
        """Convert timing stats to a dictionary."""
        return {
            "audio_extraction_seconds": self.audio_extraction_time,
            "diarization_seconds": self.diarization_time,
            "transcription_seconds": self.transcription_time,
            "average_chunk_diarization_seconds": np.mean(self.chunk_diarization_times) if self.chunk_diarization_times else 0,
            "total_chunks_processed": len(self.chunk_diarization_times),
            "average_segment_transcription_seconds": np.mean(self.segment_transcription_times) if self.segment_transcription_times else 0,
            "total_segments_transcribed": len(self.segment_transcription_times)
        }
    
    def __str__(self):
        """String representation of timing stats."""
        result = "Performance Timing Statistics:\n"
        result += "============================\n"
        result += f"Audio Extraction Time: {self.audio_extraction_time:.2f} seconds\n"
        result += f"Diarization Time: {self.diarization_time:.2f} seconds\n"
        
        if self.chunk_diarization_times:
            result += f"  - Average per chunk: {np.mean(self.chunk_diarization_times):.2f} seconds\n"
            result += f"  - Total chunks: {len(self.chunk_diarization_times)}\n"
        
        result += f"Transcription Time: {self.transcription_time:.2f} seconds\n"
        
        if self.segment_transcription_times:
            result += f"  - Average per segment: {np.mean(self.segment_transcription_times):.2f} seconds\n"
            result += f"  - Total segments: {len(self.segment_transcription_times)}\n"
        
        total_time = self.audio_extraction_time + self.diarization_time + self.transcription_time
        result += f"Total Processing Time: {total_time:.2f} seconds\n"
        
        return result

class SpeakerDiarizerWithTranscription:
    def __init__(self, hf_token=None, chunk_duration=600, overlap_duration=30, 
                 whisper_model="base", language=None, pyannote_model_path="pyannote/speaker-diarization-3.1"):
        """
        Initialize the speaker diarization processor with transcription.
        
        Args:
            hf_token (str): HuggingFace authentication token
            chunk_duration (int): Duration of each chunk in seconds (default: 10 minutes)
            overlap_duration (int): Overlap between chunks in seconds (default: 30 seconds)
            whisper_model (str): Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
            language (str): Language code for transcription (e.g., 'en', 'fr', 'de', etc.)
        """
        self.hf_token = hf_token or "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"
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.whisper_model_size = whisper_model
        self.language = language
        self.temp_dir = None
        self.pipeline = None
        self.asr_processor = None
        self.asr_model = None
        self.pyannote_model_path = pyannote_model_path
        # Store device as string for easier output, but convert to torch.device when needed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize timing statistics
        self.timing_stats = TimingStats()
    
    def initialize_pipeline(self):
        """Initialize the diarization pipeline."""
        if self.pipeline is None:
            print("Initializing diarization pipeline...")
            start_time = time.time()
            
            self.pipeline = Pipeline.from_pretrained(
                self.pyannote_model_path,
                use_auth_token=self.hf_token
            )
            # Convert string device to torch.device
            device = torch.device(self.device)
            self.pipeline = self.pipeline.to(device)
            
            load_time = time.time() - start_time
            print(f"Using {self.device.upper()} for diarization")
            print(f"Pipeline initialization took {load_time:.2f} seconds")
    
    def initialize_asr(self):
        """Initialize the ASR model."""
        if self.asr_model is None:
            print(f"Initializing Whisper {self.whisper_model_size} model for transcription...")
            start_time = time.time()
            
            model_name = f"openai/whisper-{self.whisper_model_size}"
            self.asr_processor = WhisperProcessor.from_pretrained(model_name)
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.asr_model = self.asr_model.to("cuda")
                print("Using GPU for transcription")
            else:
                print("Using CPU for transcription")
                
            load_time = time.time() - start_time
            print(f"ASR model initialization took {load_time:.2f} seconds")
    
    def create_temp_directory(self):
        """Create a temporary directory for processing chunks."""
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.temp_dir}")
        return self.temp_dir
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def extract_audio(self, video_path, output_audio_path=None):
        """
        Extract audio from video file.
        
        Args:
            video_path (str): Path to the video file
            output_audio_path (str, optional): Path for the extracted audio
            
        Returns:
            tuple: (audio_path, duration_in_seconds)
        """
        if output_audio_path is None:
            if self.temp_dir is None:
                self.create_temp_directory()
            output_audio_path = os.path.join(self.temp_dir, "full_audio.wav")
        
        print(f"Extracting audio from: {video_path}")
        start_time = time.time()
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit audio
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            output_audio_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # Calculate extraction time
        self.timing_stats.audio_extraction_time = time.time() - start_time
        print(f"Audio extraction took {self.timing_stats.audio_extraction_time:.2f} seconds")
        print(f"Audio extracted to: {output_audio_path}")
        
        # Get duration of audio
        with contextlib.closing(wave.open(output_audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            
        print(f"Audio duration: {format_timestamp(duration)}")
        return output_audio_path, duration
    
    def get_audio_duration(self, audio_path):
        """Get the duration of an audio file."""
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    
    def extract_audio_chunk(self, audio_path, start_time, end_time, output_path):
        """
        Extract a chunk of audio from the full audio file.
        
        Args:
            audio_path (str): Path to the full audio file
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            output_path (str): Path to save the chunk
            
        Returns:
            str: Path to the extracted chunk
        """
        cmd = [
            'ffmpeg',
            '-i', audio_path,
            '-ss', str(start_time),  # Start time
            '-to', str(end_time),    # End time
            '-c', 'copy',            # Copy without re-encoding
            '-y',                    # Overwrite output
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return output_path
    
    def transcribe_segment(self, audio_path, start_time, end_time):
        """
        Transcribe a specific segment of audio.
        
        Args:
            audio_path (str): Path to the full audio file
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            
        Returns:
            str: Transcribed text
        """
        # Initialize ASR if not already done
        if self.asr_model is None:
            self.initialize_asr()
            
        # Extract the segment to transcribe
        segment_path = os.path.join(self.temp_dir, f"segment_{start_time:.2f}_{end_time:.2f}.wav")
        self.extract_audio_chunk(audio_path, start_time, end_time, segment_path)
        
        try:
            # Start timing this segment's transcription
            segment_start_time = time.time()
            
            # Load audio using librosa
            audio_array, sampling_rate = librosa.load(segment_path, sr=16000)
            
            # Process through Whisper
            input_features = self.asr_processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features
            
            # Move to GPU if available
            if torch.cuda.is_available():
                input_features = input_features.to("cuda")
            
            # Generate token ids
            forced_decoder_ids = None
            if self.language:
                forced_decoder_ids = self.asr_processor.get_decoder_prompt_ids(
                    language=self.language, task="transcribe"
                )
            
            predicted_ids = self.asr_model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids
            )
            
            # Decode the token ids to text
            transcription = self.asr_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            # Record the time it took to transcribe this segment
            segment_time = time.time() - segment_start_time
            self.timing_stats.segment_transcription_times.append(segment_time)
            
            # Clean up temporary segment file
            os.remove(segment_path)
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Error transcribing segment {start_time:.2f}-{end_time:.2f}: {e}")
            return ""
    
    def process_audio_in_chunks(self, audio_path, audio_duration):
        """
        Process long audio by dividing it into overlapping chunks.
        
        Args:
            audio_path (str): Path to the full audio file
            audio_duration (float): Duration of the audio in seconds
            
        Returns:
            dict: Combined diarization results
        """
        if self.temp_dir is None:
            self.create_temp_directory()
            
        # Initialize the pipeline if not already done
        self.initialize_pipeline()
        
        # Calculate number of chunks
        num_chunks = math.ceil(audio_duration / self.chunk_duration)
        print(f"Processing audio in {num_chunks} chunks of {self.chunk_duration} seconds with {self.overlap_duration} seconds overlap")
        
        all_results = []
        diarization_start_time = time.time()  # Start timing the diarization process
        
        # Process each chunk
        for i in tqdm(range(num_chunks), desc="Processing chunks"):
            # Calculate chunk boundaries
            start_time = max(0, i * self.chunk_duration - self.overlap_duration)
            end_time = min(audio_duration, (i + 1) * self.chunk_duration + self.overlap_duration)
            
            # Skip if chunk is too short
            if end_time - start_time < 5:  # Skip chunks shorter than 5 seconds
                continue
                
            chunk_path = os.path.join(self.temp_dir, f"chunk_{i}.wav")
            
            # Extract chunk
            self.extract_audio_chunk(audio_path, start_time, end_time, chunk_path)
            
            # Apply diarization with timing
            try:
                chunk_start_time = time.time()
                
                diarization = self.pipeline(chunk_path)
                
                # Record time for this chunk's diarization
                chunk_time = time.time() - chunk_start_time
                self.timing_stats.chunk_diarization_times.append(chunk_time)
                
                # Store results with adjusted timestamps
                chunk_results = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    # Adjust timestamps relative to the full audio
                    adjusted_start = start_time + turn.start
                    adjusted_end = start_time + turn.end
                    
                    # Only keep results from the non-overlapping portion for non-edge chunks
                    if i == 0 and adjusted_end > self.chunk_duration:
                        # For first chunk, keep everything up to chunk_duration + half of overlap
                        if adjusted_start > self.chunk_duration + (self.overlap_duration / 2):
                            continue
                    elif i == num_chunks - 1 and adjusted_start < i * self.chunk_duration:
                        # For last chunk, keep everything after (i * chunk_duration - half of overlap)
                        if adjusted_end < i * self.chunk_duration - (self.overlap_duration / 2):
                            continue
                    elif i > 0 and i < num_chunks - 1:
                        # For middle chunks, only keep the central non-overlapping part
                        if adjusted_start < i * self.chunk_duration - (self.overlap_duration / 2) or \
                           adjusted_end > (i + 1) * self.chunk_duration + (self.overlap_duration / 2):
                            continue
                    
                    chunk_results.append({
                        "start": adjusted_start,
                        "end": adjusted_end,
                        "speaker": speaker
                    })
                
                all_results.extend(chunk_results)
                
                # Remove chunk file to save space
                os.remove(chunk_path)
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
        
        # Total diarization time
        self.timing_stats.diarization_time = time.time() - diarization_start_time
        print(f"Total diarization processing took {self.timing_stats.diarization_time:.2f} seconds")
        
        # Sort results by start time
        all_results.sort(key=lambda x: x["start"])
        
        # Merge adjacent segments from the same speaker with small gaps
        merged_results = self.merge_segments(all_results)
        
        return merged_results
    
    def merge_segments(self, segments, max_gap=0.5):
        """
        Merge adjacent segments from the same speaker if they're close enough.
        
        Args:
            segments (list): List of segment dictionaries
            max_gap (float): Maximum gap in seconds to merge
            
        Returns:
            list: Merged segments
        """
        if not segments:
            return []
            
        merged = [segments[0]]
        
        for segment in segments[1:]:
            last = merged[-1]
            
            # If same speaker and close enough, merge
            if segment["speaker"] == last["speaker"] and segment["start"] - last["end"] <= max_gap:
                last["end"] = segment["end"]
            else:
                merged.append(segment)
                
        return merged
    
    def add_transcriptions(self, segments, audio_path):
        """
        Add transcriptions to each segment.
        
        Args:
            segments (list): List of segment dictionaries
            audio_path (str): Path to the audio file
            
        Returns:
            list: Segments with added transcriptions
        """
        print("\nAdding transcriptions to segments...")
        
        # Initialize ASR model
        self.initialize_asr()
        
        # Start timing transcription process
        transcription_start_time = time.time()
        
        # Add transcriptions to segments
        for i, segment in enumerate(tqdm(segments, desc="Transcribing")):
            transcription = self.transcribe_segment(
                audio_path, segment["start"], segment["end"]
            )
            segment["text"] = transcription
        
        # Total transcription time
        self.timing_stats.transcription_time = time.time() - transcription_start_time
        print(f"Total transcription processing took {self.timing_stats.transcription_time:.2f} seconds")
            
        return segments
    
    def process_video(self, video_path, output_file=None, keep_audio=False, transcribe=True):
        """
        Process a video file for speaker diarization and optionally transcription.
        
        Args:
            video_path (str): Path to the video file
            output_file (str, optional): Path for the diarization results
            keep_audio (bool): Whether to keep the extracted audio file
            transcribe (bool): Whether to perform transcription
            
        Returns:
            str: Path to the output file
        """
        try:
            # Record total processing start time
            total_start_time = time.time()
            
            # Create temp directory
            self.create_temp_directory()
            
            # Extract audio
            audio_path, audio_duration = self.extract_audio(video_path)
            
            # Process audio in chunks
            diarization_results = self.process_audio_in_chunks(audio_path, audio_duration)
            
            # Add transcriptions if requested
            if transcribe:
                diarization_results = self.add_transcriptions(diarization_results, audio_path)
            
            # Determine output file path
            if output_file is None:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_file = f"{base_name}_diarization.txt"
            
            # Calculate total processing time
            total_processing_time = time.time() - total_start_time
            
            # Write results
            self.write_diarization_results(
                diarization_results, 
                output_file, 
                audio_duration, 
                transcribe,
                total_processing_time
            )
            
            # Save JSON results as well
            json_output = output_file.replace('.txt', '.json')
            
            # Add timing stats to the JSON output
            output_data = {
                "diarization_results": diarization_results,
                "timing_stats": self.timing_stats.to_dict(),
                "audio_duration": audio_duration,
                "total_processing_time": total_processing_time
            }
            
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nDiarization results saved to {output_file} and {json_output}")
            
            # Move the audio file if requested
            if keep_audio:
                final_audio_path = os.path.splitext(output_file)[0] + "_audio.wav"
                shutil.copy(audio_path, final_audio_path)
                print(f"Preserved audio file at: {final_audio_path}")
            
            return output_file
            
        except Exception as e:
            print(f"Error processing video: {e}")
            raise
            
        finally:
            # Clean up temporary files
            if not keep_audio:
                self.cleanup()
    
    def write_diarization_results(self, diarization_results, output_file, audio_duration, 
                                 with_transcription=True, total_processing_time=None):
        """
        Write diarization results to a file.
        
        Args:
            diarization_results (list): List of diarization result dictionaries
            output_file (str): Path to the output file
            audio_duration (float): Duration of the audio in seconds
            with_transcription (bool): Whether transcription is included
            total_processing_time (float): Total processing time in seconds
        """
        print(f"Writing results to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Speaker Diarization Results\n")
            f.write("==========================\n\n")
            f.write(f"Total audio duration: {format_timestamp(audio_duration)}\n\n")
            
            # Write performance statistics
            f.write("Performance Statistics:\n")
            f.write("----------------------\n")
            f.write(f"Audio extraction time: {self.timing_stats.audio_extraction_time:.2f} seconds\n")
            f.write(f"Diarization processing time: {self.timing_stats.diarization_time:.2f} seconds\n")
            
            if self.timing_stats.chunk_diarization_times:
                f.write(f"  - Average chunk processing time: {np.mean(self.timing_stats.chunk_diarization_times):.2f} seconds\n")
                f.write(f"  - Total chunks processed: {len(self.timing_stats.chunk_diarization_times)}\n")
            
            if with_transcription:
                f.write(f"Transcription processing time: {self.timing_stats.transcription_time:.2f} seconds\n")
                if self.timing_stats.segment_transcription_times:
                    f.write(f"  - Average segment transcription time: {np.mean(self.timing_stats.segment_transcription_times):.2f} seconds\n")
                    f.write(f"  - Total segments transcribed: {len(self.timing_stats.segment_transcription_times)}\n")
            
            if total_processing_time:
                f.write(f"Total processing time: {total_processing_time:.2f} seconds\n")
                f.write(f"Processing ratio: {total_processing_time / audio_duration:.2f}x realtime\n\n")
            
            # Calculate speaking time per speaker
            speaker_times = {}
            for segment in diarization_results:
                speaker = segment["speaker"]
                duration = segment["end"] - segment["start"]
                
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += duration
            
            # Print speaking time summary
            f.write("Speaking Time Summary:\n")
            f.write("---------------------\n")
            sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
            
            for speaker, total_time in sorted_speakers:
                percentage = (total_time / audio_duration) * 100
                f.write(f"{speaker}: {format_timestamp(total_time)} ({percentage:.2f}% of total)\n")
            
            f.write("\nDetailed Timeline:\n")
            f.write("----------------\n")
            
            # Write ordered segments
            for segment in diarization_results:
                start = segment["start"]
                end = segment["end"]
                speaker = segment["speaker"]
                duration = end - start
                
                f.write(f"{format_timestamp(start)} → {format_timestamp(end)} [{format_timestamp(duration)}] : {speaker}\n")
                
                # Add transcription if available
                if with_transcription and "text" in segment and segment["text"]:
                    f.write(f"    {segment['text']}\n\n")
                else:
                    f.write("\n")

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def main():
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Speaker diarization and transcription for large video files")
    
    # Required arguments
    parser.add_argument("video_path", help="Path to the input video file")
    
    # Optional arguments
    parser.add_argument("--token", "-t", dest="hf_token", 
                        help="HuggingFace token for model access")
    parser.add_argument("--pyannote_model", "-p",dest="pyannote_model_path", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--output", "-o", dest="output_file",
                        help="Path for the output diarization file")
    parser.add_argument("--chunk-size", "-c", dest="chunk_minutes", type=int, default=10,
                        help="Size of each processing chunk in minutes (default: 10)")
    parser.add_argument("--overlap", "-v", dest="overlap_seconds", type=int, default=30,
                        help="Overlap between chunks in seconds (default: 30)")
    parser.add_argument("--keep-audio", "-k", dest="keep_audio", action="store_true",
                        help="Keep the extracted audio file after processing")
    parser.add_argument("--device", "-d", dest="device", choices=["cpu", "cuda"], 
                        help="Processing device (cpu or cuda, default: auto-detect)")
    parser.add_argument("--whisper-model", "-w", dest="whisper_model", 
                        choices=["tiny", "base", "small", "medium", "large"], default="small",
                        help="Whisper model size for transcription (default: base)")
    parser.add_argument("--language", "-l", dest="language", 
                        help="Language code for transcription (e.g., 'en' for English)")
    parser.add_argument("--no-transcribe", "-n", dest="no_transcribe", action="store_true",
                        help="Skip transcription, only perform diarization")
    parser.add_argument("--timing-report", "-r", dest="timing_report", action="store_true",
                        help="Generate a detailed performance timing report")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Calculate chunk duration in seconds
    chunk_duration = args.chunk_minutes * 60
    overlap_duration = args.overlap_seconds
    
    # Create default output path if not specified
    if not args.output_file:
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        output_file = f"{base_name}_diarization.txt"
    else:
        output_file = args.output_file
    
    # Process video
    try:
        diarizer = SpeakerDiarizerWithTranscription(
            hf_token=args.hf_token,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            whisper_model=args.whisper_model,
            language=args.language,
            pyannote_model_path=args.pyannote_model_path
        )
        
        # Override device if specified
        if args.device:
            diarizer.device = args.device
            print(f"Using specified device: {args.device}")
        
        diarizer.process_video(
            video_path=args.video_path, 
            output_file=output_file,
            keep_audio=args.keep_audio,
            transcribe=not args.no_transcribe
        )
        
        # Print timing summary
        print("\nPerformance Timing Summary:")
        print(diarizer.timing_stats)
        
        # Generate additional timing report if requested
        if args.timing_report:
            timing_file = os.path.splitext(output_file)[0] + "_timing_report.txt"
            with open(timing_file, 'w', encoding='utf-8') as f:
                f.write(str(diarizer.timing_stats))
                
                # Add detailed chunk timings
                if diarizer.timing_stats.chunk_diarization_times:
                    f.write("\nDetailed Chunk Processing Times (seconds):\n")
                    for i, time_val in enumerate(diarizer.timing_stats.chunk_diarization_times):
                        f.write(f"Chunk {i}: {time_val:.2f}\n")
                
                # Add detailed segment timings
                if diarizer.timing_stats.segment_transcription_times:
                    f.write("\nDetailed Segment Transcription Times (seconds):\n")
                    for i, time_val in enumerate(diarizer.timing_stats.segment_transcription_times):
                        f.write(f"Segment {i}: {time_val:.2f}\n")
            
            print(f"Detailed timing report saved to: {timing_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()