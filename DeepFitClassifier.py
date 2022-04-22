import tensorflow as tf
import numpy as np


# def load_X(X_path):
#     file = open(X_path, "r")
#     X_ = np.array([elem for elem in [row.split(",") for row in file]], dtype=np.float32)
#     file.close()
#     return X_


def euclidean_dist(a, b):
    # This function calculates the euclidean distance between 2 point in 2-D coordinates
    # if one of two points is (0,0), dist = 0
    # a, b: input array with dimension: m, 2
    # m: number of samples
    # 2: x and y coordinate
    try:
        if a.shape[1] == 2 and a.shape == b.shape:
            # check if element of a and b is (0,0)
            bol_a = (a[:, 0] != 0).astype(int)
            bol_b = (b[:, 0] != 0).astype(int)
            dist = np.linalg.norm(a - b, axis=1)
            return (dist * bol_a * bol_b).reshape(a.shape[0], 1)
    except:
        print("[Error]: Check dimension of input vector")
        return 0


def norm_X(X):
    num_sample = X.shape[0]
    # Keypoints
    Nose = X[:, 0 * 2 : 0 * 2 + 2]
    Neck = X[:, 1 * 2 : 1 * 2 + 2]
    RShoulder = X[:, 2 * 2 : 2 * 2 + 2]
    RElbow = X[:, 3 * 2 : 3 * 2 + 2]
    RWrist = X[:, 4 * 2 : 4 * 2 + 2]
    LShoulder = X[:, 5 * 2 : 5 * 2 + 2]
    LElbow = X[:, 6 * 2 : 6 * 2 + 2]
    LWrist = X[:, 7 * 2 : 7 * 2 + 2]
    RHip = X[:, 8 * 2 : 8 * 2 + 2]
    RKnee = X[:, 9 * 2 : 9 * 2 + 2]
    RAnkle = X[:, 10 * 2 : 10 * 2 + 2]
    LHip = X[:, 11 * 2 : 11 * 2 + 2]
    LKnee = X[:, 12 * 2 : 12 * 2 + 2]
    LAnkle = X[:, 13 * 2 : 13 * 2 + 2]
    REye = X[:, 14 * 2 : 14 * 2 + 2]
    LEye = X[:, 15 * 2 : 15 * 2 + 2]
    REar = X[:, 16 * 2 : 16 * 2 + 2]
    LEar = X[:, 17 * 2 : 17 * 2 + 2]

    # Length of head
    length_Neck_LEar = euclidean_dist(Neck, LEar)
    length_Neck_REar = euclidean_dist(Neck, REar)
    length_Neck_LEye = euclidean_dist(Neck, LEye)
    length_Neck_REye = euclidean_dist(Neck, REye)
    length_Nose_LEar = euclidean_dist(Nose, LEar)
    length_Nose_REar = euclidean_dist(Nose, REar)
    length_Nose_LEye = euclidean_dist(Nose, LEye)
    length_Nose_REye = euclidean_dist(Nose, REye)
    length_head = np.maximum.reduce(
        [
            length_Neck_LEar,
            length_Neck_REar,
            length_Neck_LEye,
            length_Neck_REye,
            length_Nose_LEar,
            length_Nose_REar,
            length_Nose_LEye,
            length_Nose_REye,
        ]
    )
    # length_head      = np.sqrt(np.square((LEye[:,0:1]+REye[:,0:1])/2 - Neck[:,0:1]) + np.square((LEye[:,1:2]+REye[:,1:2])/2 - Neck[:,1:2]))

    # Length of torso
    length_Neck_LHip = euclidean_dist(Neck, LHip)
    length_Neck_RHip = euclidean_dist(Neck, RHip)
    length_torso = np.maximum(length_Neck_LHip, length_Neck_RHip)
    # length_torso     = np.sqrt(np.square(Neck[:,0:1]-(LHip[:,0:1]+RHip[:,0:1])/2) + np.square(Neck[:,1:2]-(LHip[:,1:2]+RHip[:,1:2])/2))

    # Length of right leg
    length_leg_right = euclidean_dist(RHip, RKnee) + euclidean_dist(RKnee, RAnkle)
    # length_leg_right = np.sqrt(np.square(RHip[:,0:1]-RKnee[:,0:1]) + np.square(RHip[:,1:2]-RKnee[:,1:2])) \
    # + np.sqrt(np.square(RKnee[:,0:1]-RAnkle[:,0:1]) + np.square(RKnee[:,1:2]-RAnkle[:,1:2]))

    # Length of left leg
    length_leg_left = euclidean_dist(LHip, LKnee) + euclidean_dist(LKnee, LAnkle)
    # length_leg_left = np.sqrt(np.square(LHip[:,0:1]-LKnee[:,0:1]) + np.square(LHip[:,1:2]-LKnee[:,1:2])) \
    # + np.sqrt(np.square(LKnee[:,0:1]-LAnkle[:,0:1]) + np.square(LKnee[:,1:2]-LAnkle[:,1:2]))

    # Length of leg
    length_leg = np.maximum(length_leg_right, length_leg_left)

    # Length of body
    length_body = length_head + length_torso + length_leg

    # Check all samples have length_body of 0
    length_chk = (length_body > 0).astype(int)

    # Check keypoints at origin
    keypoints_chk = (X > 0).astype(int)

    chk = length_chk * keypoints_chk

    # Set all length_body of 0 to 1 (to avoid division by 0)
    length_body[length_body == 0] = 1

    # The center of gravity
    # number of point OpenPose locates:
    num_pts = (X[:, 0::2] > 0).sum(1).reshape(num_sample, 1)
    centr_x = X[:, 0::2].sum(1).reshape(num_sample, 1) / num_pts
    centr_y = X[:, 1::2].sum(1).reshape(num_sample, 1) / num_pts

    # The  coordinates  are  normalized relative to the length of the body and the center of gravity
    X_norm_x = (X[:, 0::2] - centr_x) / length_body
    X_norm_y = (X[:, 1::2] - centr_y) / length_body

    # Stack 1st element x and y together
    X_norm = np.column_stack((X_norm_x[:, :1], X_norm_y[:, :1]))

    for i in range(1, X.shape[1] // 2):
        X_norm = np.column_stack(
            (X_norm, X_norm_x[:, i : i + 1], X_norm_y[:, i : i + 1])
        )

    # Set all samples have length_body of 0 to origin (0, 0)
    X_norm = X_norm * chk

    return X_norm


# encodings for labels in dataset
LABELS = [
    "squats",
    "lunges",
    "bicep_curls",
    "situps",
    "pushups",
    "tricep_extensions",
    "dumbbell_rows",
    "jumping_jacks",
    "dumbbell_shoulder_press",
    "lateral_shoulder_raises",
]
class_d = {}
for i, ex in enumerate(LABELS):
    class_d[ex] = i


class DeepFitClassifier:
    def __init__(self, model_path):
        """
        Load the TFLite model and allocate tensors \n
        Get input and output tensors.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, X) -> str:
        """
        X -> (1, 36) shaped numpy array
        First 18 are the x coordinates of the 18 keypoints \n
        next 18 are the y coordinates of the 18 keypoints \n
        Returns: \n
        Winning Label
        """
        # Setup input
        X_sample = np.array(X).reshape(1, -1)
        X_sample_norm = norm_X(X_sample)
        input_data = np.array(X_sample_norm[0], np.float32).reshape(1, 36)
        # Invoke the model on the input data
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        # Get the result
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

        np_output_data = np.array(output_data)
        champ_idx = np.argmax(np_output_data)
        self.results = dict(zip(LABELS, output_data[0]))

        return LABELS[champ_idx]

    def get_results(self) -> dict:
        """
        Get a dictionary containing the probability of each label for the last predicted input
        """
        return self.results
        # return 0

