import requests
import json

# API endpoint
url = "http://localhost:8000/process-video"

# Test video file
video_path = "data/videos/2025-05-22_08-25-12_UTC.mp4"

# Caption and hashtags
caption = "We started with a feeling."
hashtags = [
    "Now it has a name.",
    "This is Beyond The Curve.",
    "Not a trend. Not a label.",
    "A new standard, built from scratch, shaped by every woman who was told to wait."
]

# Prepare the files and data
files = {
    'video': ('video.mp4', open(video_path, 'rb'), 'video/mp4')
}

data = {
    'caption': caption,
    'hashtags': json.dumps(hashtags)
}

# Make the request
print("Sending request to:", url)
print("Caption:", caption)
print("Hashtags:", hashtags)
print("\nSending video for processing...")

response = requests.post(url, files=files, data=data)

# Print the response
if response.status_code == 200:
    print("\nSuccess! Response:")
    print(json.dumps(response.json(), indent=2))
else:
    print("\nError:", response.status_code)
    print(response.text) 