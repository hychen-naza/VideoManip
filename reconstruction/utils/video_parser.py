#!/usr/bin/env python3
"""
Video Parser Script
Extracts RGB images from a video file with configurable frame rate and output options.
"""

import cv2
import os
import argparse
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoParser:
    def __init__(self, video_path, output_dir="extracted_frames", frame_rate=1, quality=95, rotate=False):
        """
        Initialize the video parser.
        
        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save extracted frames
            frame_rate (int): Extract every Nth frame (1 = all frames, 2 = every 2nd frame, etc.)
            quality (int): JPEG quality for saved images (1-100)
            rotate (bool): Whether to rotate frames 90 degrees to the right
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.frame_rate = frame_rate
        self.quality = quality
        self.rotate = rotate
        self.cap = None
        
    def extract_frames(self):
        """Extract frames from the video and save as RGB images."""
        try:
            # Open video file
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video_path}")
            
            # Get video properties
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video properties:")
            logger.info(f"  - Total frames: {total_frames}")
            logger.info(f"  - FPS: {fps:.2f}")
            logger.info(f"  - Resolution: {width}x{height}")
            logger.info(f"  - Duration: {total_frames/fps:.2f} seconds")
            
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {self.output_dir}")
            
            # Extract frames
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Extract every Nth frame based on frame_rate
                if frame_count % self.frame_rate == 0:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # upside down the frame
                    # rgb_frame = cv2.flip(rgb_frame, 0)
                    # # right side left the frame
                    # rgb_frame = cv2.flip(rgb_frame, 1)
                    # Rotate frame if requested
                    if self.rotate:
                        rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
                    
                    # Save frame
                    frame_filename = f"frame_{saved_count:06d}.png"
                    frame_path = self.output_dir / frame_filename
                    
                    # Save as JPEG with specified quality
                    cv2.imwrite(str(frame_path), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), 
                               [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    
                    saved_count += 1
                    
                    if saved_count % 100 == 0:
                        logger.info(f"Saved {saved_count} frames...")
                
                frame_count += 1
            
            logger.info(f"Extraction complete! Saved {saved_count} frames to {self.output_dir}")
            
            # Create a summary file
            self._create_summary(total_frames, saved_count, fps, width, height)
            
        except Exception as e:
            logger.error(f"Error during frame extraction: {e}")
            raise
        finally:
            if self.cap:
                self.cap.release()
    
    def _create_summary(self, total_frames, saved_count, fps, width, height):
        """Create a summary file with extraction details."""
        summary_path = self.output_dir / "extraction_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Video Frame Extraction Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Input video: {self.video_path}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Frame extraction rate: every {self.frame_rate} frame(s)\n")
            f.write(f"JPEG quality: {self.quality}\n")
            if self.rotate:
                f.write("Frame rotation: 90 degrees to the right\n")
            f.write(f"Video properties:\n")
            f.write(f"  - Total frames: {total_frames}\n")
            f.write(f"  - FPS: {fps:.2f}\n")
            f.write(f"  - Resolution: {width}x{height}\n")
            f.write(f"  - Duration: {total_frames/fps:.2f} seconds\n\n")
            f.write(f"Extraction results:\n")
            f.write(f"  - Frames extracted: {saved_count}\n")
            f.write(f"  - Extraction ratio: {saved_count/total_frames:.2%}\n")
            f.write(f"  - Effective FPS: {fps/self.frame_rate:.2f}\n")
        
        logger.info(f"Summary saved to: {summary_path}")
    
    def get_video_info(self):
        """Get basic information about the video without extracting frames."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps
            
            cap.release()
            
            return {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Extract RGB frames from a video file')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('-o', '--output', default='extracted_frames', 
                       help='Output directory for extracted frames (default: extracted_frames)')
    parser.add_argument('-r', '--rate', type=int, default=1,
                       help='Extract every Nth frame (default: 1 = all frames)')
    parser.add_argument('-q', '--quality', type=int, default=95,
                       help='JPEG quality (1-100, default: 95)')
    parser.add_argument('--info-only', action='store_true',
                       help='Only display video information without extracting frames')
    parser.add_argument('--rotate', action='store_true',
                       help='Rotate frames 90 degrees to the right')
    
    args = parser.parse_args()
    print(args.rotate)
    # Check if video file exists
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return
    
    

    # Create video parser
    video_parser = VideoParser(
        video_path=args.video_path,
        output_dir=args.output,
        frame_rate=args.rate,
        quality=args.quality,
        rotate=args.rotate
    )
    
    # Get video information
    info = video_parser.get_video_info()
    if info:
        logger.info("Video Information:")
        logger.info(f"  - Total frames: {info['total_frames']}")
        logger.info(f"  - FPS: {info['fps']:.2f}")
        logger.info(f"  - Resolution: {info['width']}x{info['height']}")
        logger.info(f"  - Duration: {info['duration']:.2f} seconds")
        
        if args.rate > 1:
            estimated_frames = info['total_frames'] // args.rate
            logger.info(f"  - Estimated frames to extract: {estimated_frames}")
    
    # Extract frames if not info-only mode
    if not args.info_only:
        logger.info("Starting frame extraction...")
        video_parser.extract_frames()

if __name__ == "__main__":
    main() 