import cv2
import mediapipe as mp
import numpy as np
import math



cap = cv2.VideoCapture(0)
count = 0
direction = 0
form = 0
feedback = "Bad Form. Correct Posture."



def set_pose_parameters():
    mode = False 
    complexity = 1
    smooth_landmarks = True
    enable_segmentation = False
    smooth_segmentation = True
    detectionCon = 0.5
    trackCon = 0.5
    
    mpPose = mp.solutions.pose
    return mode,complexity,smooth_landmarks,enable_segmentation,smooth_segmentation,detectionCon,trackCon,mpPose


mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
pose = mpPose.Pose(mode, complexity, smooth_landmarks,
                                enable_segmentation, smooth_segmentation,
                                detectionCon, trackCon)

def get_pose (img, results, draw=True):        
        if results.pose_landmarks:
            if draw:
                mpDraw = mp.solutions.drawing_utils
                mpDraw.draw_landmarks(img,results.pose_landmarks,
                                           mpPose.POSE_CONNECTIONS) 
        return img

def get_position(img, draw=True):
        landmark_list = []
        if results.pose_landmarks:
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                #finding height, width of the image printed
                height, width, c = img.shape
                #Determining the pixels of the landmarks
                landmark_pixel_x, landmark_pixel_y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, landmark_pixel_x, landmark_pixel_y])
                if draw:
                    cv2.circle(img, (landmark_pixel_x, landmark_pixel_y), 5, (255,0,0), cv2.FILLED)
        return landmark_list    


def get_angle(img, landmark_list, point1, point2, point3, draw=True):   
        #Retrieve landmark coordinates from point identifiers
        x1, y1 = landmark_list[point1][1:]
        x2, y2 = landmark_list[point2][1:]
        x3, y3 = landmark_list[point3][1:]
        
        
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
        
        #Handling angle edge cases: Obtuse and negative angles
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle
        
        
        if draw:
            #Drawing lines between the three points
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)

            #Drawing circles at intersection points of lines
            cv2.circle(img, (x1, y1), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (75,0,130), 2)
            cv2.circle(img, (x2, y2), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (75,0,130), 2)
            cv2.circle(img, (x3, y3), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (75,0,130), 2)
            
            #Show angles between lines
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        return angle
    
    
    
while cap.isOpened():
    #Getting image from camera
    ret, img = cap.read() 
    #Getting video dimensions
    width  = cap.get(3)  
    height = cap.get(4)  
    
    #Convert from BGR (used by cv2) to RGB (used by Mediapipe)
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    #Get pose and draw landmarks
    img = get_pose(img, results, False)
    
    landmark_list = get_position(img, False)
    #If landmarks exist, get the elbow, shoulder and hip. The points used are identifiers for specific joints
    if len(landmark_list) != 0:
        elbow_angle = get_angle(img, landmark_list, 11, 13, 15)
        shoulder_angle = get_angle(img, landmark_list, 13, 11, 23)
        hip_angle = get_angle(img, landmark_list, 11, 23,25)
        

        pushup_success_percentage = np.interp(elbow_angle, (90, 160), (0, 100))
        

        pushup_progress_bar = np.interp(elbow_angle, (90, 160), (380, 50))

        #Is the form correct at the start?
        if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
            form = 1
    
        #Full pushup motion
        if form == 1:
            if pushup_success_percentage == 0:
                if elbow_angle <= 90 and hip_angle > 160:
                    feedback = "Go Up"
                    if direction == 0:
                        count += 0.5
                        direction = 1
                else:
                    feedback = "Bad Form. Correct Posture."
                    
            if pushup_success_percentage == 100:
                if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
                    feedback = "Go Down"
                    if direction == 1:
                        count += 0.5
                        direction = 0
                else:
                    feedback = "Bad Form. Correct Posture."
                
        print(count)
        
        #Draw the pushup progress bar
        if form == 1:
            cv2.rectangle(img, (580, 50), (600, 380), (0, 255, 0), 3)
            cv2.rectangle(img, (580, int(pushup_progress_bar)), (600, 380), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(pushup_success_percentage)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)


        #Show the pushup count
        cv2.rectangle(img, (0, 380), (100, 480), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5)
        
        #Show the pushup feedback 
        cv2.rectangle(img, (500, 0), (640, 40), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, feedback, (500, 40 ), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0), 2)

        
    cv2.imshow('Pushup Counter', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()