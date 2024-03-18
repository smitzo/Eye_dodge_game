import pygame
import random
import numpy as np
import cv2
import os
import pyautogui
import mediapipe as mp
import threading

# Initialize pygame
pygame.init()

# Set up the screen
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Eye Dodge")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Load the Haar cascade for detecting eyes
cascade_path = os.path.join(os.getcwd(), "haarcascade_eye.xml")
cascade = cv2.CascadeClassifier(cascade_path)

# Initialize the video capture device
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open video capture.")

# Initialize face mesh for eye control
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Player attributes
PLAYER_WIDTH, PLAYER_HEIGHT = 50, 50
player_x = SCREEN_WIDTH // 2 - PLAYER_WIDTH // 2
player_y = SCREEN_HEIGHT - PLAYER_HEIGHT - 20
player_speed = 5

# Obstacle attributes
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 50, 50
obstacle_speed = 3  # Decreased speed
obstacle_freq = 50  # Lower values increase frequency
obstacles = []

# Game variables
score = 0
font = pygame.font.Font(None, 36)

# Function to detect and process eyes
def scan(image_size=(32, 32)):
    _, frame = video_capture.read()
    frame = cv2.resize(frame, (320, 240))  # Decrease frame resolution
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, 1.1, 5)  # Adjust detection parameters
    if len(boxes) == 2:
        eyes = []
        for box in boxes:
            x, y, w, h = box
            eye = frame[y:y + h, x:x + w]
            eye = cv2.resize(eye, image_size)
            eye = normalize(eye)
            eye = eye[10:-10, 5:-5]
            eyes.append((x, y, w, h))  # Store eye coordinates
        return frame, eyes
    else:
        return None, None

# Function for face landmark detection
def detect_landmarks():
    while True:
        _, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape
        if landmark_points:
            landmarks = landmark_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))
                if id == 1:
                    screen_x = screen_w * landmark.x *1.02
                    screen_y = screen_h * landmark.y *1.02
                    pyautogui.moveTo(screen_x, screen_y)

                    # Set player's x-coordinate based on eye position
                    global player_x
                    player_x = int(x * (SCREEN_WIDTH - PLAYER_WIDTH) / frame_w)
                    
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if (left[0].y - left[1].y) < 0.004:
                print("eyes clicked")
                pyautogui.click()
                
                pyautogui.sleep(1)
        #cv2.imshow("Eye Tracker", frame)
        cv2.waitKey(1)

# Function to normalize the image
def normalize(image):
    return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 

# Start face landmark detection in a separate thread
landmark_thread = threading.Thread(target=detect_landmarks)
landmark_thread.start()

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(BLACK)
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw player
    pygame.draw.rect(screen, WHITE, (player_x, player_y, PLAYER_WIDTH, PLAYER_HEIGHT))
    
    # Spawn obstacles
    if random.randrange(0, obstacle_freq) == 0:
        obstacle_x = random.randint(0, SCREEN_WIDTH - OBSTACLE_WIDTH)
        obstacle_y = 0 - OBSTACLE_HEIGHT
        obstacles.append([obstacle_x, obstacle_y])
    
    # Move obstacles
    for obstacle in obstacles:
        obstacle[1] += obstacle_speed
        pygame.draw.rect(screen, RED, (obstacle[0], obstacle[1], OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        
        # Collision detection
        if (player_x < obstacle[0] + OBSTACLE_WIDTH and
            player_x + PLAYER_WIDTH > obstacle[0] and
            player_y < obstacle[1] + OBSTACLE_HEIGHT and
            player_y + PLAYER_HEIGHT > obstacle[1]):
            print("Game Over!")
            running = False
    
    # Remove obstacles that have passed the screen
    obstacles = [obstacle for obstacle in obstacles if obstacle[1] < SCREEN_HEIGHT]
    
    # Update score
    score += 1
    
    # Display score
    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (10, 10))
    
    pygame.display.flip()
    clock.tick(60)

# Release the video capture device and close all windows
video_capture.release()
cv2.destroyAllWindows()
pygame.quit()
