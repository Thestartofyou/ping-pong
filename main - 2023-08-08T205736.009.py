import cv2
import numpy as np

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize variables for ball position and movement history
ball_position = None
ball_history = []

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to isolate moving objects (like the ball)
    fgmask = fgbg.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            ball_position = (x + w // 2, y + h // 2)
            ball_history.append(ball_position)

    # Display ball position
    if ball_position:
        cv2.circle(frame, ball_position, 10, (0, 255, 0), -1)

    # Display frame
    cv2.imshow("Ping Pong Ball Tracking", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
