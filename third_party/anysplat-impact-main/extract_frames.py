import cv2
import os

def extract_frames_per_second(video_path, output_folder):
    """
    Extract frames from video at 1-second intervals.
    
    Args:
        video_path: Path to the input video file
        output_folder: Directory to save extracted frames
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 4)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    frame_count = 0
    saved_count = 0
    frame_name = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Calculate current time in seconds
        current_time = frame_count / fps
        
        # Check if we're at a 1-second interval
        if frame_count % int(frame_interval) == 0:
            # Time in seconds (rounded to avoid floating point issues)
            time_sec = int(round(current_time))
            time_sec = frame_name
            frame_name += 1
            
            # Save frame with time as filename
            output_path = os.path.join(output_folder, f"{time_sec}.jpg")
            cv2.imwrite(output_path, frame)
            
            saved_count += 1
            print(f"Saved frame at {time_sec}s -> {output_path}")
        
        frame_count += 1
    
    # Release resources
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"Total frames saved: {saved_count}")

# Example usage
if __name__ == "__main__":
    video_path = "/home/lauretta/quang/AnySplat/test_images_1/IMG_0478.MOV"  # Change this to your video path
    output_folder = "test_images_1"  # Change this to your desired output folder
    
    extract_frames_per_second(video_path, output_folder)