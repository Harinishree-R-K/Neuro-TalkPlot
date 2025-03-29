import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import speech_recognition as sr
import queue
import threading
import time
import subprocess
import sys
import cv2
from datetime import datetime

class SpeechToGCodeProcessor:
    def __init__(self, ugs_path=None):
        self.text_queue = queue.Queue()
        self.is_running = True
        self.ugs_path = ugs_path or self._find_ugs_path()
        self.batch_text = ""
        self.batch_threshold = 20
        self.font_size = 200
        self.font = None
        self.a4_width_mm = 210
        self.a4_height_mm = 297
        self.scale_factor = 0.08
        self.img_width = int(self.a4_width_mm / self.scale_factor)
        self.img_height = int(self.a4_height_mm / self.scale_factor)
        self.current_position = (10, 10)
        self.word_spacing = 30
        self.line_spacing = 50
        self.max_y_position = 10
        self.processing_lock = threading.Lock()  # Add lock for thread safety

    def _find_ugs_path(self):
        possible_paths = [
            r"C:\\Program Files\\Universal-G-Code-Sender\\UniversalGcodeSender.jar",
            r"C:\\Program Files (x86)\\Universal-G-Code-Sender\\UniversalGcodeSender.jar",
            "/usr/local/bin/UniversalGcodeSender.jar",
            "/opt/UniversalGcodeSender/UniversalGcodeSender.jar"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def real_time_transcription(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=2)
                print("Listening...")
                while self.is_running:
                    try:
                        audio = recognizer.listen(source, timeout=5)
                        text = recognizer.recognize_google(audio).strip()
                        if text:
                            self.batch_text += " " + text
                            print(f"Recognized: {text}")
                            if len(self.batch_text.split()) >= self.batch_threshold:
                                self.text_queue.put(self.batch_text.strip())
                                self.batch_text = ""
                    except sr.WaitTimeoutError:
                        if self.batch_text:
                            self.text_queue.put(self.batch_text.strip())
                            self.batch_text = ""
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError as e:
                        print(f"Speech recognition error: {e}")
        except Exception as e:
            print(f"Error in transcription: {e}")
            self.is_running = False

    def _load_font(self):
        handwriting_fonts = [
            "DancingScript-Regular.ttf",
            "Pacifico.ttf",
            "Gabriola.ttf"
        ]
        font_dirs = [
            r"C:\Windows\Fonts",
            r"/usr/share/fonts",
            r"/Library/Fonts",
            r"/System/Library/Fonts",
            os.path.expanduser("~/.fonts")
        ]
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for font_name in handwriting_fonts:
                    try:
                        font_path = os.path.join(font_dir, font_name)
                        if os.path.exists(font_path):
                            self.font = ImageFont.truetype(font_path, self.font_size)
                            print(f"Using font: {font_path}")
                            return
                    except Exception:
                        continue
        try:
            self.font = ImageFont.truetype("arial.ttf", self.font_size)
            print("Using Arial font")
        except IOError:
            try:
                self.font = ImageFont.truetype("DejaVuSans.ttf", self.font_size)
                print("Using DejaVuSans font")
            except IOError:
                self.font = ImageFont.load_default()
                print("Warning: Using default font, text quality may be reduced")

    def text_to_gcode(self, text, feed_rate=800, safe_z=5, cutting_z=-0.5):
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output_{timestamp}.nc"
        
        img = Image.new('RGB', (self.img_width, self.img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        if not self.font:
            self._load_font()
        
        words = text.split()
        current_x = self.current_position[0]
        current_y = self.current_position[1]
        
        for word in words:
            try:
                bbox = draw.textbbox((current_x, current_y), word, font=self.font)
                word_width = bbox[2] - bbox[0]
                word_height = bbox[3] - bbox[1]
            except AttributeError:
                word_width, word_height = self.font.getsize(word)
            
            if current_x + word_width > self.img_width:
                current_x = 10
                current_y += self.font_size + self.line_spacing
            
            draw.text((current_x, current_y), word, font=self.font, fill=(0, 0, 0))
            current_x += word_width + self.word_spacing
            self.max_y_position = max(self.max_y_position, current_y + word_height)
        
        if current_x + self.word_spacing > self.img_width:
            self.current_position = (10, self.max_y_position + self.line_spacing)
        else:
            self.current_position = (current_x, current_y)
        
        temp_img_path = f"temp_text_{timestamp}.png"
        img.save(temp_img_path)
        
        cv_img = cv2.imread(temp_img_path)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        contours_result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
        
        with open(output_file, 'w') as f:
            f.write(f"G21 ; Set units to millimeters\n")
            f.write(f"G90 ; Set absolute positioning\n")
            f.write(f"G0 Z{safe_z} ; Move to safe height\n")
            f.write(f"G0 X0 Y0 ; Move to origin\n")
            
            for contour in contours:
                if len(contour) > 2:
                    epsilon = 0.0005 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    start_x = approx_contour[0][0][0] * self.scale_factor
                    start_y = (self.img_height - approx_contour[0][0][1]) * self.scale_factor
                    f.write(f"G0 Z{safe_z}\n")
                    f.write(f"G0 X{start_x:.3f} Y{start_y:.3f}\n")
                    f.write(f"G1 Z{cutting_z} F{feed_rate}\n")
                    
                    for point in approx_contour[1:]:
                        x = point[0][0] * self.scale_factor
                        y = (self.img_height - point[0][1]) * self.scale_factor
                        f.write(f"G1 X{x:.3f} Y{y:.3f} F{feed_rate}\n")
                    f.write(f"G0 Z{safe_z}\n")
            
            f.write(f"G0 Z{safe_z}\n")
            f.write(f"G0 X0 Y0\n")
            f.write("M5\n")
            f.write("M30\n")
            
        try:
            os.remove(temp_img_path)
        except:
            pass
            
        print(f"G-code saved to {output_file}")
        return output_file

    def send_to_ugs(self, gcode_file):
        if not self.ugs_path:
            print("UGS not found.")
            return False
        
        try:
            gcode_absolute_path = os.path.abspath(gcode_file)
            if not os.path.exists(gcode_absolute_path):
                print(f"Error: G-code file {gcode_absolute_path} not found")
                return False
            
            if self.ugs_path.endswith('.jar'):
                subprocess.Popen(["java", "-jar", self.ugs_path, "--open", gcode_absolute_path])
            else:
                subprocess.Popen([self.ugs_path, "--open", gcode_absolute_path, "--console", "new"])  # Open new console
                
            print(f"Sent G-code to UGS: {gcode_absolute_path}")
            return True
        except Exception as e:
            print(f"Error sending to UGS: {e}")
            return False

    def process_queue(self):
        while self.is_running:
            if not self.text_queue.empty():
                with self.processing_lock:  # Ensure only one batch is processed at a time
                    text = self.text_queue.get().strip()
                    if text:
                        print(f"Processing: {text}")
                        gcode_file = self.text_to_gcode(text)
                        if gcode_file:
                            self.send_to_ugs(gcode_file)
                            time.sleep(2)  # Wait for UGS to launch before processing next batch
            time.sleep(0.1)

    def run(self):
        try:
            threading.Thread(target=self.real_time_transcription, daemon=True).start()
            threading.Thread(target=self.process_queue, daemon=True).start()
            print("Press Ctrl+C to stop.")
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.is_running = False
            print("Shutting down...")

if __name__ == "__main__":
    processor = SpeechToGCodeProcessor("C:\\Users\\K_har\\Downloads\\win64-ugs-platform-app-2.1.12\\ugsplatform-win\\bin\\ugsplatform64.exe")
    processor.run()