import requests
import json
import os
import sys

def test_video_processing(video_path: str, caption: str = "", hashtags: list = None):
    """Test the video processing pipeline by sending a video to the API."""
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    print(f"\nVideo file exists: {video_path}")
    print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    # Prepare the request
    url = "http://localhost:8000/process-video"
    files = {
        'video': ('video.mp4', open(video_path, 'rb'), 'video/mp4')
    }
    data = {
        'caption': caption,
        'hashtags': json.dumps(hashtags) if hashtags else None
    }
    
    print(f"\nSending request to: {url}")
    print(f"Caption: {caption}")
    print(f"Hashtags: {hashtags}")
    
    # Send the request
    try:
        print("\nSending video for processing...")
        response = requests.post(url, files=files, data=data)
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            return
            
        # Print the results
        result = response.json()
        print("\nResults:")
        print(json.dumps(result, indent=2))
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server. Make sure it's running at http://localhost:8000")
    except requests.exceptions.RequestException as e:
        print(f"Error processing video: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding response: {e}")
        print(f"Raw response: {response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        files['video'][1].close()

if __name__ == "__main__":
    # Test with a sample video
    video_path = "data/videos/2025-05-22_08-25-12_UTC.mp4"
    
    # Read caption and hashtags from the corresponding txt file
    txt_path = video_path.replace('.mp4', '.txt')
    caption = ""
    hashtags = []
    
    if os.path.exists(txt_path):
        print(f"Reading caption and hashtags from: {txt_path}")
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            if lines:
                caption = lines[0].strip()
                hashtags = [line.strip() for line in lines[1:] if line.strip()]
                print(f"Found caption: {caption}")
                print(f"Found hashtags: {hashtags}")
    else:
        print(f"Warning: No caption file found at {txt_path}")
    
    test_video_processing(video_path, caption, hashtags) 