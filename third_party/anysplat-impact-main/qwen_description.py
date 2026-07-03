import numpy as np
import cv2
import torch
import json
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


class VideoAnalyzer:
    """
    Video analysis class using Qwen2-VL model.
    Analyzes videos for environment, people count, activities, threats, and anomalies.
    """
    
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", question_prompt=None):
        """
        Initialize the VideoAnalyzer with model and prompt.
        
        Args:
            model_name (str): HuggingFace model name
            question_prompt (str): Custom question prompt for analysis
        """
        print(f"Loading model: {model_name}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Default prompt if none provided
        self.question_prompt = question_prompt or """Analyze this video and provide the following information:
1. Environment: Describe the setting/location. Provide only the environment type without any additional commentary.
2. Number of people: Count how many people are visible in the video
3. Activities: Describe what the people are doing (if any people are present)
4. Threats: Identify any potential threats to human life or safety. Just list the threats without any additional commentary. If no threats are visible, state "No threats detected".
5. Is there anomaly: Answer TRUE if anything unusual, unexpected, or out of place is happening in the video. Answer FALSE if everything appears normal.
6. Anomaly description: If there is an anomaly (answer to question 5 is TRUE), describe what the anomaly is in one sentence. If no anomaly, write "None".

Please be specific and detailed in your analysis."""
        
        print("Model loaded successfully!")
    
    def load_video(self, video_path, sampling=16):
        """
        Load video frames from file.
        
        Args:
            video_path (str): Path to video file
            sampling (int): Frame sampling rate (process every Nth frame)
            
        Returns:
            np.ndarray: Array of video frames
        """
        cap = cv2.VideoCapture(video_path)
        ret_frames = []
        frame_idx = 0
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx % sampling) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ret_frames.append(frame)
            frame_idx += 1
        
        cap.release()
        
        if len(ret_frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        print(f"Loaded {len(ret_frames)} frames from video")
        return np.stack(ret_frames)
    
    def analyze_video(self, video_path, sampling=8, max_tokens=512):
        """
        Analyze video and return raw model output.
        
        Args:
            video_path (str): Path to video file
            sampling (int): Frame sampling rate
            max_tokens (int): Maximum tokens for generation
            
        Returns:
            str: Raw model output text
        """
        frames = self.load_video(video_path, sampling)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": self.question_prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            videos=[frames],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        print("Generating analysis...")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        
        output_text = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return output_text
    
    def parse_analysis(self, raw_output):
        """
        Parse raw model output into structured data.
        
        Args:
            raw_output (str): Raw text from model
            
        Returns:
            dict: Structured analysis with keys: environment, number_of_people, 
                  activities, threats, is_anomaly, anomaly_reason, raw_output
        """
        analysis = {
            "environment": "",
            "number_of_people": 0,
            "activities": "",
            "threats": "",
            "is_anomaly": False,
            "anomaly_reason": None,
            "raw_output": raw_output
        }
        
        lines = raw_output.lower()
        
        # Extract environment
        for line in raw_output.split('\n'):
            if any(keyword in line.lower() for keyword in ["environment", "setting", "location"]):
                analysis["environment"] = line.split(':', 1)[-1].strip() if ':' in line else line.strip()
                break
        
        # Extract number of people
        for line in raw_output.split('\n'):
            if "people" in line.lower() and any(keyword in line.lower() for keyword in ["number", "count", "there are", "there is"]):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    analysis["number_of_people"] = int(numbers[0])
                    break
                elif any(word in line.lower() for word in ["no people", "zero", "none", "empty", "no one"]):
                    analysis["number_of_people"] = 0
                    break
        
        # Extract activities
        for line in raw_output.split('\n'):
            if any(keyword in line.lower() for keyword in ["activities", "activity", "doing", "action"]):
                analysis["activities"] = line.split(':', 1)[-1].strip() if ':' in line else line.strip()
                break
        
        # Extract threats
        for line in raw_output.split('\n'):
            if any(keyword in line.lower() for keyword in ["threat", "safety", "hazard", "danger", "risk"]):
                threat_text = line.split(':', 1)[-1].strip() if ':' in line else line.strip()
                if any(word in threat_text.lower() for word in ["none", "no threat", "no danger", "not visible", "no apparent", "safe", "no hazard"]):
                    analysis["threats"] = "No threats detected"
                else:
                    analysis["threats"] = threat_text
                break
        
        # Extract is_anomaly (question 5)
        for line in raw_output.split('\n'):
            if any(keyword in line.lower() for keyword in ["is there anomaly", "anomaly:", "is anomaly", "anomalies:"]):
                line_lower = line.lower()
                if "true" in line_lower or "yes" in line_lower:
                    analysis["is_anomaly"] = True
                elif "false" in line_lower or "no" in line_lower:
                    analysis["is_anomaly"] = False
                break
        
        # Extract anomaly_reason (question 6)
        for line in raw_output.split('\n'):
            if any(keyword in line.lower() for keyword in ["anomaly description", "anomaly reason", "describe the anomaly", "what is the anomaly"]):
                anomaly_desc = line.split(':', 1)[-1].strip() if ':' in line else line.strip()
                if anomaly_desc and not any(word in anomaly_desc.lower() for word in ["none", "no anomaly", "n/a", "not applicable"]):
                    analysis["anomaly_reason"] = anomaly_desc
                else:
                    analysis["anomaly_reason"] = None
                break
        
        return analysis
    
    def __call__(self, video_path, sampling=16, max_tokens=512, return_raw=False):
        """
        Analyze video (callable interface).
        
        Args:
            video_path (str): Path to video file
            sampling (int): Frame sampling rate
            max_tokens (int): Maximum tokens for generation
            return_raw (bool): If True, return both structured and raw output
            
        Returns:
            dict: Structured analysis results
        """
        raw_output = self.analyze_video(video_path, sampling, max_tokens)
        structured_output = self.parse_analysis(raw_output)
        
        if return_raw:
            return structured_output, raw_output
        return structured_output
    
    def set_prompt(self, new_prompt):
        """
        Update the question prompt.
        
        Args:
            new_prompt (str): New prompt text
        """
        self.question_prompt = new_prompt
        print("Prompt updated!")


# Example usage
if __name__ == "__main__":
    # Initialize analyzer with default prompt (using smaller 2B model)
    analyzer = VideoAnalyzer()
    
    # Or with custom prompt
    # custom_prompt = "Describe what you see in this video in detail."
    # analyzer = VideoAnalyzer(question_prompt=custom_prompt)
    
    # Analyze a video
    video_path = "/home/lauretta/quang/impact/AnySplat_impact/video_output_path/rgb.mp4"
    
    # Method 1: Using call
    result = analyzer(video_path)
    
    # Method 2: Get both structured and raw output
    # result, raw_text = analyzer(video_path, return_raw=True)
    
    # Display results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS:")
    print("="*60)
    print(json.dumps(result, indent=2))
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Environment: {result['environment']}")
    print(f"Number of People: {result['number_of_people']}")
    print(f"Activities: {result['activities']}")
    print(f"Threats: {result['threats']}")
    print(f"Is Anomaly: {result['is_anomaly']}")
    print(f"Anomaly Reason: {result['anomaly_reason']}")
    
    # Send to API (matches your API schema)
    import requests
    
    api_data = {
        "environment": result['environment'],
        "description": result['activities'],
        "number_of_people": result['number_of_people'],
        "threats": result['threats'] if result['threats'] != "No threats detected" else None,
        "is_anomaly": result['is_anomaly'],
        "anomaly_reason": result['anomaly_reason']
    }
    
    # Uncomment to send to your server
    # response = requests.post('http://localhost:3001/api/update-analysis', json=api_data)
    # print(f"\nAPI Response: {response.json()}")