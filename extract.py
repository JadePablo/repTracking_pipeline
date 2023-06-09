#not packaged with python (install dependencies before use)
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
#packaged with python
import time
import winsound
from thresholds import threshold_dict

#mediapipe modules for pose detection and custom frame overlay
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.

    Args:
        a (numpy.array): First point coordinates [x, y].
        b (numpy.array): Mid point coordinates [x, y].
        c (numpy.array): End point coordinates [x, y].

    Returns:
        float: The calculated angle in degrees.
    """
    a = np.array(a)  # indicator 1
    b = np.array(b)  # intersection of indicators
    c = np.array(c)  # indicator 2

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def convergingExercise_extract(exercise_name,targeted_reps):
    """
    Extract data for a converging exercise.

    Args:
        exercise_name (str): The name of the exercise.
        targeted_reps (int): The number of reps to aim for.

    Returns:
        dict: Extracted exercise data and exercise name.
    """

    exercise_df = pd.DataFrame(columns=['angle','time'])

    bottom = threshold_dict[exercise_name][0]
    top = threshold_dict[exercise_name][1]

    reps = 0
    last_stage = "bottom"

    print('press key to get in position and start recording')
    input()
    print('timer started')

    for remaining in range(10,0,-1):
        print(f"{remaining} seconds to get in position")
        time.sleep(1)


    start_time = time.time()



    cap = cv2.VideoCapture(0)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        #make noise to indicate start of recording
        winsound.Beep(440, 1000)

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                #get time
                elapsed_time = time.time() - start_time


                #update rep-counter here using thresholds

                if angle > bottom:
                    #started at bottom, do nothing
                    #came from top, count the rep.
                    if last_stage == "top":
                        reps += 1
                    last_stage = "bottom"

                if angle < top:
                    last_stage = "top"


                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                #record angle and time of recording
                entry = pd.DataFrame({
                    'angle': angle,
                    'time': elapsed_time
                },index = [len(exercise_df)])
                exercise_df = pd.concat([exercise_df,entry],ignore_index=True)

            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # Render rep counter
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(reps),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            #optional kill condition when counted_reps = targeted_reps
            if (reps >= targeted_reps):
                break

        cap.release()
        cv2.destroyAllWindows()

        #make noise to indicate that recording has stopped.
        winsound.Beep(440, 1000)

        return {'data': exercise_df, 'name': exercise_name}


def divergingExercise_extract(exercise_name,targeted_reps):
    """
    Extract data for a diverging exercise.

    Args:
        exercise_name (str): The name of the exercise.
        targeted_reps (int): The number of reps to aim for.

    Returns:
        dict: Extracted exercise data and exercise name.
    """
    exercise_df = pd.DataFrame(columns=['angle','time'])

    bottom = threshold_dict[exercise_name][0]
    top = threshold_dict[exercise_name][1]

    reps = 0
    last_stage = "bottom"

    print('press key to get in position and start recording')
    input()
    print('timer started')

    for remaining in range(10,0,-1):
        print(f"{remaining} seconds to get in position")
        time.sleep(1)


    start_time = time.time()



    cap = cv2.VideoCapture(0)
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        #make the noise here
        winsound.Beep(440, 1000)

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                #get time
                elapsed_time = time.time() - start_time


                #update rep-counter here using thresholds
                #ur at the bottom of hte movement
                if angle < bottom:
                    #started at bottom, do nothing
                    #came from top, count the rep.
                    last_stage = "bottom"

                if angle > top:
                    if last_stage == "bottom":
                        reps += 1
                    last_stage = "top"


                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                #record angle and time of recording
                entry = pd.DataFrame({
                    'angle': angle,
                    'time': elapsed_time
                },index = [len(exercise_df)])
                exercise_df = pd.concat([exercise_df,entry],ignore_index=True)

            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # Render rep counter here
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(reps),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            #add optional kill condition when counted_reps = targeted_reps
            if (reps >= targeted_reps):
                break

        cap.release()
        cv2.destroyAllWindows()

        #make noise here to indicate that recording has stopped.
        winsound.Beep(440, 1000)

        return {'data': exercise_df, 'name': exercise_name}


def prompt_exercise() -> dict:
    print('curls')

    exercise = input('pick an exercise: choose from above or i\'ll keep asking: ')

    while (exercise not in threshold_dict):
        exercise = input('pick an exercise: choose from above or i\'ll keep asking: ')

    reps = int(input('how many reps are you aiming for: '))

    while reps <= 0:
        reps = input('stop bullshitting me: ')


    return {
        'reps': reps,
        'exercise': exercise
    }

def extract():
    input = prompt_exercise()

    exercise_group = threshold_dict[input['exercise']][2]
    print(exercise_group)
    if (exercise_group == "converge"):
        return convergingExercise_extract(input['exercise'],input['reps'])

    if(exercise_group == "diverge"):
        return divergingExercise_extract(input['exercise'],input['reps'])