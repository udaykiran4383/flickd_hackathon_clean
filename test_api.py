import requests
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api_endpoint(url: str, video_path: str, caption: str = None, hashtags: list = None) -> Dict[str, Any]:
    """Test the API endpoint with a video file."""
    try:
        # Prepare the files and data
        files = {
            'video': ('video.mp4', open(video_path, 'rb'), 'video/mp4')
        }
        
        data = {}
        if caption:
            data['caption'] = caption
        if hashtags:
            data['hashtags'] = json.dumps(hashtags)
            
        # Make the request
        response = requests.post(url, files=files, data=data)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            logger.info("API test successful")
            logger.info(f"Response: {json.dumps(result, indent=2)}")
            return result
        else:
            logger.error(f"API test failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error testing API: {str(e)}")
        return None

if __name__ == "__main__":
    # Test configuration
    API_URL = "http://localhost:8000/process_video"
    VIDEO_PATH = "data/videos/test_video.mp4"
    CAPTION = "Summer vibes with my new outfit! #fashion #summer"
    HASHTAGS = ["#fashion", "#summer", "#outfit"]
    
    # Run test
    result = test_api_endpoint(API_URL, VIDEO_PATH, CAPTION, HASHTAGS) 