# Install necessary libraries
!pip install transformers flask flask-ngrok pyngrok torch torchaudio soundfile moviepy diffusers
!pip install git+https://github.com/facebookresearch/audiocraft.git

# Install ngrok for public access
!pip install pyngrok

from pyngrok import ngrok

# Authenticate ngrok with your authtoken
NGROK_AUTH_TOKEN = "2rzyfW8sqC2vnhmrWYq0UbuXGyX_3h9xgyJJGBFGK96W4ywLs"  # Replace with your actual ngrok auth token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Import necessary libraries for video and music generation
from audiocraft.models import MusicGen
import torchaudio
import moviepy.editor as mp
from flask import Flask, request, jsonify, send_file
from diffusers import DiffusionPipeline
import torch

# Initialize MusicGen model
print("Loading MusicGen model...")
model = MusicGen.get_pretrained("melody")

# Function to generate music
def generate_music(genre="pop", duration=10, output_file="generated_music.wav"):
    """
    Generate music using the MusicGen model.
    :param genre: str: Genre of music (used as text prompt).
    :param duration: int: Duration of the music in seconds.
    :param output_file: str: Path to save the generated music.
    """
    print(f"Generating {duration}s of {genre} music...")
    model.set_generation_params(duration=duration)  # Set the duration in seconds
    music = model.generate([genre], progress=True)  # Generate music based on genre

    # Save the generated music to a file
    torchaudio.save(output_file, music[0].cpu(), sample_rate=32000)
    print(f"Music saved at: {output_file}")

# Initialize LTX-Video pipeline for text-to-video generation
print("Loading LTX-Video model...")
pipe = DiffusionPipeline.from_pretrained("Lightricks/LTX-Video")

# Function to generate video from text prompt
def generate_video_from_text(video_prompt, num_frames, output_file):
    """
    Generate video using LTX-Video from the given text prompt and duration.
    :param video_prompt: str: The prompt to generate video
    :param num_frames: int: Number of frames to generate for the video
    :param output_file: str: Path to save the final video
    """
    print(f"Generating video for prompt: '{video_prompt}' with {num_frames} frames.")
    try:
        frames = []
        for i in range(num_frames):
            frame = pipe(video_prompt).images[0]  # Generate a single frame
            frames.append(frame)
            print(f"Generated frame {i+1}/{num_frames}")

        # Combine frames into a video using moviepy
        clip = mp.ImageSequenceClip([frame for frame in frames], fps=24)  # Assuming 24 fps for video
        clip.write_videofile(output_file, codec="libx264")
        print(f"Video generated and saved at: {output_file}")
    except Exception as e:
        print(f"Error generating video: {e}")
        raise

# Function to combine video and audio
def combine_video_audio(video_file, audio_file, output_file):
    """
    Combine generated video and audio into a final video output.
    """
    video = mp.VideoFileClip(video_file)
    audio = mp.AudioFileClip(audio_file)

    # Set the audio of the video clip
    final_video = video.set_audio(audio)

    # Write the final video with audio to a file
    final_video.write_videofile(output_file, codec="libx264")
    print(f"Final video saved at: {output_file}")

# Initialize Flask app for API
app = Flask(__name__)

# Start ngrok and get the public URL
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

@app.route('/generate_content', methods=['POST'])
def generate_content():
    data = request.get_json()

    # Extract user input from the request
    genre = data.get("genre", "pop")  # Default genre is 'pop'
    duration = int(data.get("duration", 10))  # Default duration is 10 seconds
    video_prompt = data.get("video_prompt", "A peaceful scenic view of nature.")  # Default video prompt

    # Generate music first
    music_output_file = "generated_music.wav"
    try:
        generate_music(genre=genre, duration=duration, output_file=music_output_file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Calculate the number of frames for the video (assuming 24 fps)
    num_frames = duration * 24  # 24 frames per second
    # Generate video
    video_output_file = "generated_video.mp4"
    try:
        generate_video_from_text(video_prompt, num_frames, video_output_file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Combine the generated music and video into a final video file
    combined_output_file = "final_video.mp4"
    combine_video_audio(video_output_file, music_output_file, combined_output_file)

    # Return the final video to the user
    return send_file(combined_output_file, as_attachment=True)

if __name__ == "__main__":
    app.run(port=5000)
