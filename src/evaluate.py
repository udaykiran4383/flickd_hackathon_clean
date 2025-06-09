import os
import json
from datetime import datetime
import sys

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import FlickdPipeline

def evaluate_videos(video_dir, output_dir):
    """
    Process all videos in the directory and generate evaluation JSONs.
    
    Args:
        video_dir (str): Directory containing input videos
        output_dir (str): Directory to save evaluation JSONs
    """
    # Initialize pipeline
    pipeline = FlickdPipeline()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each video
    for video_file in os.listdir(video_dir):
        if not video_file.endswith('.mp4'):
            continue
            
        print(f"Processing {video_file}...")
        
        # Generate output filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S_UTC")
        output_file = os.path.join(output_dir, f"{timestamp}.json")
        
        try:
            # Process video
            video_path = os.path.join(video_dir, video_file)
            result = pipeline.process_video(video_path)
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue

if __name__ == "__main__":
    # Directories
    VIDEO_DIR = "data/videos"
    OUTPUT_DIR = "outputs"
    
    # Run evaluation
    evaluate_videos(VIDEO_DIR, OUTPUT_DIR) 