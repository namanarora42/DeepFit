import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from DeepFitClassifier import DeepFitClassifier
import numpy as np
from collections import deque
import math


# Mapping dictionary to map keypoints from Mediapipe to our Classifier model
lm_dict = {
  0:0 , 1:10, 2:12, 3:14, 4:16, 5:11, 6:13, 7:15, 8:24, 9:26, 10:28, 11:23, 12:25, 13:27, 14:5, 15:2, 16:8, 17:7,
}



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


def get_pose (img, results, draw=True):        
        if results.pose_landmarks:
            if draw:
                mpDraw = mp.solutions.drawing_utils
                mpDraw.draw_landmarks(img,results.pose_landmarks,
                                           mpPose.POSE_CONNECTIONS) 
        return img

def get_position(img, results, height, width, draw=True ):
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

    
    
def convert_mediapipe_keypoints_for_model(lm_dict, landmark_list):
    inp_pushup = []
    for index in range(0, 36):
        if index < 18:
            inp_pushup.append(round(landmark_list[lm_dict[index]][1],3))
        else:
            inp_pushup.append(round(landmark_list[lm_dict[index-18]][2],3))
    return inp_pushup



# Setting variables for video feed
def set_video_feed_variables():
    cap = cv2.VideoCapture(0)
    count = 0
    direction = 0
    form = 0
    feedback = "Bad Form."
    frame_queue = deque(maxlen=150)
    clf = DeepFitClassifier('deepfit_classifier_v3.tflite')
    return cap,count,direction,form,feedback,frame_queue,clf


def set_percentage_bar_and_text(elbow_angle):
    pushup_success_percentage = np.interp(elbow_angle, (90, 160), (0, 100))
    pushup_progress_bar = np.interp(elbow_angle, (90, 160), (380, 30))
    return pushup_success_percentage,pushup_progress_bar

def set_body_angles_from_keypoints(get_angle, img, landmark_list):
    elbow_angle = get_angle(img, landmark_list, 11, 13, 15)
    shoulder_angle = get_angle(img, landmark_list, 13, 11, 23)
    hip_angle = get_angle(img, landmark_list, 11, 23,25)
    return elbow_angle,shoulder_angle,hip_angle

def set_smoothened_workout_name(lm_dict, convert_mediapipe_keypoints_for_model, frame_queue, clf, landmark_list):
    inp_pushup = convert_mediapipe_keypoints_for_model(lm_dict, landmark_list)
    workout_name = clf.predict(inp_pushup)
    frame_queue.append(workout_name)
    workout_name_after_smoothening = max(set(frame_queue), key=frame_queue.count)
    return "Workout Name: " + workout_name_after_smoothening

def run_full_workout_motion(count, direction, form, elbow_angle, shoulder_angle, hip_angle, pushup_success_percentage, feedback, workout_name_after_smoothening):
    if workout_name_after_smoothening == "pushups":
        if form == 1:
            if pushup_success_percentage == 0:
                if elbow_angle <= 90 and hip_angle > 160:
                    feedback = "Feedback: Go Up"
                    if direction == 0:
                        count += 0.5
                        direction = 1
                else:
                    feedback = "Feedback: Bad Form."
                        
            if pushup_success_percentage == 100:
                if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
                    feedback = "Feedback: Go Down"
                    if direction == 1:
                        count += 0.5
                        direction = 0
                else:
                    feedback = "Feedback: Bad Form."
        return [feedback, count]
    # For now, else condition handles just squats
    else:
        if form == 1:
            if pushup_success_percentage == 0:
                if elbow_angle <= 90 and hip_angle > 160:
                    feedback = "Go Up"
                    if direction == 0:
                        count += 0.5
                        direction = 1
                else:
                    feedback = "Bad Form."
                    
            if pushup_success_percentage == 100:
                if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
                    feedback = "Go Down"
                    if direction == 1:
                        count += 0.5
                        direction = 0
                else:
                    feedback = "Bad Form."
            return [feedback, count]

def draw_percentage_progress_bar(form, img, pushup_success_percentage, pushup_progress_bar):
    xd, yd, wd, hd = 10, 175, 50, 200
    if form == 1:
        cv2.rectangle(img, (xd,30), (xd+wd, yd+hd), (0, 255, 0), 3)
        cv2.rectangle(img, (xd, int(pushup_progress_bar)), (xd+wd, yd+hd), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(pushup_success_percentage)}%', (xd, yd+hd+50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)

def display_rep_count(count, img):
    xc, yc = 85, 100
    cv2.putText(img, "Reps: " + str(int(count)), (xc, yc), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

def show_workout_feedback(feedback, img):    
    xf, yf = 85, 70
    cv2.putText(img, feedback, (xf, yf), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 255, 255), 2)

def show_workout_name_from_model(img, workout_name_after_smoothening):
    xw, yw = 85, 40
    cv2.putText(img, workout_name_after_smoothening, (xw,yw), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 255, 255), 2)

def check_form(elbow_angle, shoulder_angle, hip_angle, form, workout_name_after_smoothening):
    if workout_name_after_smoothening == "pushups":
        if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
            form = 1
    # For now, else impleements squats condition        
    else:
        if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
            form = 1
    return form

def display_workout_stats(count, form, feedback, draw_percentage_progress_bar, display_rep_count, show_workout_feedback, show_workout_name_from_model, img, pushup_success_percentage, pushup_progress_bar, workout_name_after_smoothening):
    #Draw the pushup progress bar
    draw_percentage_progress_bar(form, img, pushup_success_percentage, pushup_progress_bar)

    #Show the rep count
    display_rep_count(count, img)
        
    #Show the pushup feedback 
    show_workout_feedback(feedback, img)
        
    #Show workout name
    show_workout_name_from_model(img, workout_name_after_smoothening)




def main():
    mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
    pose = mpPose.Pose(mode, complexity, smooth_landmarks,
                                enable_segmentation, smooth_segmentation,
                                detectionCon, trackCon)


    # Setting video feed variables
    cap, count, direction, form, feedback, frame_queue, clf = set_video_feed_variables()



    #Start video feed and run workout
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
        
        # Get landmark list from mediapipe
        landmark_list = get_position(img, results, height, width, False)
        
        #If landmarks exist, get the relevant workout body angles and run workout. The points used are identifiers for specific joints
        if len(landmark_list) != 0:
            elbow_angle, shoulder_angle, hip_angle = set_body_angles_from_keypoints(get_angle, img, landmark_list)
            
            pushup_success_percentage, pushup_progress_bar = set_percentage_bar_and_text(elbow_angle)
        
            workout_name_after_smoothening = set_smoothened_workout_name(lm_dict, convert_mediapipe_keypoints_for_model, frame_queue, clf, landmark_list)    
                    
            #Is the form correct at the start?
            form = check_form(elbow_angle, shoulder_angle, hip_angle, form, workout_name_after_smoothening)
        
            #Full workout motion
            if (run_full_workout_motion(count, direction, form, elbow_angle, shoulder_angle, hip_angle, pushup_success_percentage, feedback, workout_name_after_smoothening)!= None):
                feedback, count = run_full_workout_motion(count, direction, form, elbow_angle, shoulder_angle, hip_angle, pushup_success_percentage, feedback, workout_name_after_smoothening)
            
            
            
            
            #Display workout stats        
            display_workout_stats(count, form, feedback, draw_percentage_progress_bar, display_rep_count, show_workout_feedback, show_workout_name_from_model, img, pushup_success_percentage, pushup_progress_bar, workout_name_after_smoothening)
            
            
            # Transparent Overlay
            overlay = img.copy()
            x, y, w, h = 75, 10, 500, 150
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)      
            alpha = 0.4  # Transparency factor.
            # Following line overlays transparent rectangle over the image
            image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)          
            
        cv2.imshow('DEEPFIT Workout Trainer', image_new)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()