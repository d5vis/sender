import cv2
import math
import sys
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans

class Sender:
    def __init__(self, file_path=None):
        """
        Initialize the Sender class
        
        Parameters:
        - file_path (str): The path to the video file
        """
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path) if self.file_path else cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose()

        self.previous_cog = None
        self.speed = 0
        self.speed_threshold = 0.03
        self.direction = None

        self.holds = []
        self.color = (0, 0, 0)
        self.color_lower = None
        self.color_upper = None
        self.moves = 0

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.blue = (251,168,53)
        self.orange = (4,180,236)

    def center_of_gravity(self, lm):
        """
        Calculate the center of gravity
        
        Parameters:
        - lm (mediapipe.solutions.pose.PoseLandmarkList): The pose landmarks
        
        Returns:
        - tuple: The center of gravity
        """
        total_x = 0
        total_y = 0
        num_points = 0
        for point in lm.landmark:
            if point:
                total_x += point.x
                total_y += point.y
                num_points += 1
        if num_points > 0:
            return (total_x / num_points, total_y / num_points)
        else:
            return None
    
    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
    def angle(self, x1, y1, x2, y2, x3, y3):
        """
        Calculate the angle between three points
        
        Parameters:
        - x(n) (float): The x-coordinate of the point
        - y(n) (float): The y-coordinate of the point
            
        Returns:
        - float: The angle between the three points
        """
        v1 = (x2 - x1, y2 - y1)
        v2 = (x3 - x2, y3 - y2)
        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag_v1 = math.sqrt(sum(x**2 for x in v1))
        mag_v2 = math.sqrt(sum(x**2 for x in v2))
        if mag_v1 * mag_v2 == 0:
            return 0
        cosine_theta = dot_product / (mag_v1 * mag_v2)
        radians = math.acos(cosine_theta)
        degrees = math.degrees(radians) % 360
        return round(degrees)
    
    def shoulder_angles(self, lm):
        """
        Calculate the angles of the arms
        
        Parameters:
        - lm (mediapipe.solutions.pose.PoseLandmarkList): The pose landmarks

        Returns:
        - list: The angles of the arms
        """
        left_shoulder = lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        return [self.angle(left_shoulder.x, left_shoulder.y, left_elbow.x, left_elbow.y, left_wrist.x, left_wrist.y), self.angle(right_shoulder.x, right_shoulder.y, right_elbow.x, right_elbow.y, right_wrist.x, right_wrist.y)]
    
    def leg_angles(self, lm):
        """
        Calculate the angles of the legs

        Parameters:
        - lm (mediapipe.solutions.pose.PoseLandmarkList): The pose landmarks

        Returns:
        - list: The angles of the legs
        """
        left_hip = lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        right_hip = lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
        return [self.angle(left_hip.x, left_hip.y, left_knee.x, left_knee.y, left_ankle.x, left_ankle.y), self.angle(right_hip.x, right_hip.y, right_knee.x, right_knee.y, right_ankle.x, right_ankle.y)]
    
    def select_holds(self, frame):
        """
        Select the holds in the frame
        
        Parameters:
        - frame (numpy.ndarray): The frame
        
        Returns:
        - list: The holds
        """
        def nothing():
            pass

        # Create a window to display the adjusted threshold values
        cv2.namedWindow('Sender - Select Holds', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sender - Select Holds', 300, 300)
        # Create trackbars for the HSV range
        cv2.createTrackbar('HMin', 'Sender - Select Holds', 0, 179, nothing)
        cv2.createTrackbar('HMax', 'Sender - Select Holds', 0, 179, nothing)
        cv2.createTrackbar('SMin', 'Sender - Select Holds', 0, 255, nothing)
        cv2.createTrackbar('SMax', 'Sender - Select Holds', 0, 255, nothing)
        cv2.createTrackbar('VMin', 'Sender - Select Holds', 0, 255, nothing)
        cv2.createTrackbar('VMax', 'Sender - Select Holds', 0, 255, nothing)
        cv2.putText(frame, 'Select the hold color Sender - Select Holds values', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Press q to continue', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Set initial values for the trackbars
        cv2.setTrackbarPos('HMin', 'Sender - Select Holds', 0)
        cv2.setTrackbarPos('HMax', 'Sender - Select Holds', 10)
        cv2.setTrackbarPos('SMin', 'Sender - Select Holds', 164)
        cv2.setTrackbarPos('SMax', 'Sender - Select Holds', 255)
        cv2.setTrackbarPos('VMin', 'Sender - Select Holds', 0)
        cv2.setTrackbarPos('VMax', 'Sender - Select Holds', 255)

        while True:
            # Get current trackbar positions
            h_min = cv2.getTrackbarPos('HMin', 'Sender - Select Holds')
            h_max = cv2.getTrackbarPos('HMax', 'Sender - Select Holds')
            s_min = cv2.getTrackbarPos('SMin', 'Sender - Select Holds')
            s_max = cv2.getTrackbarPos('SMax', 'Sender - Select Holds')
            v_min = cv2.getTrackbarPos('VMin', 'Sender - Select Holds')
            v_max = cv2.getTrackbarPos('VMax', 'Sender - Select Holds')
            # Display the threshold values
            hsv_min = np.array([h_min, s_min, v_min])
            hsv_max = np.array([h_max, s_max, v_max])
            rgb_min = cv2.cvtColor(np.uint8([[hsv_min]]), cv2.COLOR_HSV2BGR)[0][0]
            rgb_max = cv2.cvtColor(np.uint8([[hsv_max]]), cv2.COLOR_HSV2BGR)[0][0]
            cv2.circle(frame, (50, 50), 50, (int(rgb_min[0]), int(rgb_min[1]), int(rgb_min[2])), -1)
            cv2.circle(frame, (50, 150), 50, (int(rgb_max[0]), int(rgb_max[1]), int(rgb_max[2])), -1)
            # Convert the image to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Create a mask based on the current HSV range
            self.color_lower = np.array([h_min, s_min, v_min])
            self.color_upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
            output = cv2.bitwise_and(frame, frame, mask=mask)
            # Show the masked image
            # cv2.imshow('Sender - Select Holds', frame)
            cv2.imshow('Sender - Select Holds', output)
            # If the user presses the 'q' key, break the loop
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return self.find_holds(frame, self.color_lower, self.color_upper)
    
    def find_holds(self, frame, color_lower, color_upper):
        """
        Find the holds in the frame
        
        Parameters:
        - frame (numpy.ndarray): The frame
        
        Returns:
        - list: The holds
        """
        # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Create a mask based on the color range
        mask = cv2.inRange(hsv_frame, np.array(color_lower), np.array(color_upper))
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate through contours and extract positions
        hold_positions = []
        for contour in contours:
            # Calculate centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                hold_positions.append((cx, cy))
        # return hold_positions
        # Perform K-means clustering
        hold_positions_np = np.array(hold_positions)
        kmeans = KMeans(n_clusters=15, n_init=10)
        kmeans.fit(hold_positions_np)
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_.astype(int)
        return cluster_centers.tolist()

    def draw_holds(self, frame, holds):
        """
        Draw the holds on the frame
        
        Parameters:
        - frame (numpy.ndarray): The frame
        - holds (list): The holds
        """
        for hold in holds:
            cv2.circle(frame, hold, 5, self.blue, -1)

    def read(self):
        """
        Read the video stream and process the pose landmarks
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            self.holds = self.select_holds(frame) if not self.holds else self.find_holds(frame, self.color_lower, self.color_upper)
            self.draw_holds(frame, self.holds)
    
            keypoints = self.pose.process(frame)
            lm = keypoints.pose_landmarks
            if lm:
                # TODO
                # self.update_moves(lm, self.holds)
                arm_angles = self.shoulder_angles(lm)
                cv2.putText(frame, f'Left Arm: {arm_angles[0]:.2f} degrees', (50, 50), self.font, 1, self.blue, 2, cv2.LINE_AA)
                cv2.putText(frame, f'Right Arm: {arm_angles[1]:.2f} degrees', (50, 75), self.font, 1, self.blue, 2, cv2.LINE_AA)
                leg_angles = self.leg_angles(lm)
                cv2.putText(frame, f'Left Leg: {leg_angles[0]:.2f} degrees', (50, 100), self.font, 1, self.blue, 2, cv2.LINE_AA)
                cv2.putText(frame, f'Right Leg: {leg_angles[1]:.2f} degrees', (50, 125), self.font, 1, self.blue, 2, cv2.LINE_AA)

                cog = self.center_of_gravity(lm)
                if cog:
                    cv2.line(frame, (int(cog[0] * frame.shape[1]), int(cog[1] * frame.shape[0]) + 100), (int(cog[0] * frame.shape[1]), int(cog[1] * frame.shape[0]) - 100), self.orange, 2)
                    cv2.circle(frame, (int(cog[0] * frame.shape[1]), int(cog[1] * frame.shape[0])), 5, self.blue, -1)
                    if self.previous_cog:
                        self.direction = "Up" if cog[1] < self.previous_cog[1] else "Down"
                        cv2.putText(frame, f'Direction: {self.direction}', (50, 150), self.font, 1, self.blue, 2, cv2.LINE_AA)

                        self.speed = self.distance(cog[0], cog[1], self.previous_cog[0], self.previous_cog[1])
                        if self.speed > self.speed_threshold and self.direction == "Down":
                            cv2.putText(frame, "Falling", (50, 175), self.font, 1, self.orange, 2, cv2.LINE_AA)

                    self.previous_cog = cog
                        
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(frame, lm, mp.solutions.pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=self.orange, thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=self.blue, thickness=2, circle_radius=2))

            if not self.file_path:
                fps = str(self.cap.get(cv2.CAP_PROP_FPS))
                cv2.putText(frame, fps, (50, 200), self.font, 1, self.blue, 2, cv2.LINE_AA)
            cv2.imshow(f'Sender - {"Live" if self.file_path is None else self.file_path}', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        
args = sys.argv
sender = Sender(args[1]) if len(args) > 1 else Sender()
sender.read()