import numpy as np
import tensorflow as tf
from mediapipe.python.solutions import face_mesh as mp_faces


def centerCropSquare(img, center, side=None, scaleWRTHeight=None):
    a = side is None
    b = scaleWRTHeight is None
    assert (not a and b) or (a and not b)
    half = 0
    if side is None:
        half = int(img.shape[0] * scaleWRTHeight / 2)
    else:
        half = int(side / 2)

    return img[(center[0] - half):(center[0] + half), (center[1] - half):(center[1] + half), :]

def localToGlobal(lms, center, side=None, scaleWRTHeight=None):
    a = side is None
    b = scaleWRTHeight is None
    assert (not a and b) or (a and not b)
    half = 0
    if side is None:
        half = int(img.shape[0] * scaleWRTHeight / 2)
    else:
        half = int(side / 2)

    return img[(center[0] - half):(center[0] + half), (center[1] - half):(center[1] + half), :]

from torchvision import transforms
import os
from utils.renderer import Renderer
from img2pose import img2poseModel
from model_loader import load_model
import cv2

np.set_printoptions(suppress=True)

renderer = Renderer(
    vertices_path="pose_references/vertices_trans.npy",
    triangles_path="pose_references/triangles.npy"
)

threed_points = np.load('pose_references/reference_3d_68_points_trans.npy')

transform = transforms.Compose([transforms.ToTensor()])

DEPTH = 18
MAX_SIZE = 3072
MIN_SIZE = 600

POSE_MEAN = "models/WIDER_train_pose_mean_v1.npy"
POSE_STDDEV = "models/WIDER_train_pose_stddev_v1.npy"
MODEL_PATH = "models/img2pose_v1.pth"

pose_mean = np.load(POSE_MEAN)
pose_stddev = np.load(POSE_STDDEV)

img2pose_model = img2poseModel(
    DEPTH, MIN_SIZE, MAX_SIZE,
    pose_mean=pose_mean, pose_stddev=pose_stddev,
    threed_68_points=threed_points,
)
load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
img2pose_model.evaluate()

images_path = "../../images"

threshold = 0.75

image_intrinsics = np.array([[1897, 0, 1536], [0, 1897, 864], [0, 0, 1]])

folder = "../digital-signage-interactive/frames/alexSh/3_1/2/"
for file in os.listdir(folder):

    orig_image = cv2.imread(folder + file)
    # TODO: crop automatically
    image = orig_image[:750, 1250:1750, :] #1, 3
    #image = image[600:800, 1400:1600, :] #5, 7
    #image = image[700:850, 1350:1600, :]  # 5, 7
    #image = image[750:1250, 1600:2250, :]
    #image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))

    res = img2pose_model.predict([transform(orig_image)])[0]

    all_bboxes = res["boxes"].cpu().numpy().astype('float')

    poses = []
    bboxes = []
    for i in range(len(all_bboxes)):
        if res["scores"][i] > threshold:
            bbox = all_bboxes[i]
            pose_pred = res["dofs"].cpu().numpy()[i].astype('float')
            pose_pred = pose_pred.squeeze()

            pitch_gt, yaw_gt, roll_gt = pose_pred[0], pose_pred[1], pose_pred[2]
            poses.append(pose_pred)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 500)
            fontScale = 2
            fontColor = (255, 0, 255)
            lineType = 2

            cv2.putText(orig_image, 'yaw: {}, pitch {}'.format(int(yaw_gt * 180 / np.pi), int(pitch_gt * 180 / np.pi)),
                        (50, 50),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            x, y, z = pose_pred[3:] / image_intrinsics[0, 0] * 100

    #TODO: tf model instead of mp_faces
    faces = mp_faces.FaceMesh(
            static_image_mode=True, min_detection_confidence=0.5)

    results = faces.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    multi_face_landmarks = []
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
           x = [landmark.x for landmark in landmarks.landmark]
           y = [landmark.y for landmark in landmarks.landmark]
           face_landmarks = np.transpose(np.stack((y, x))) * image.shape[0:2]
           multi_face_landmarks.append(face_landmarks)

        faces.close()
        center_y = int(multi_face_landmarks[0][159][0] + (multi_face_landmarks[0][145][0] - multi_face_landmarks[0][159][0]) / 2)
        center_x = int(multi_face_landmarks[0][130][1] + (multi_face_landmarks[0][133][1] - multi_face_landmarks[0][130][1]) / 2)

        center_y = int(multi_face_landmarks[0][386][0] + (multi_face_landmarks[0][374][0] - multi_face_landmarks[0][386][0]) / 2)
        center_x = int(multi_face_landmarks[0][362][1] + (multi_face_landmarks[0][263][1] - multi_face_landmarks[0][362][1]) / 2)

        interpreter = tf.lite.Interpreter(model_path="../digital-signage-interactive/digital_signage_demo/analytics/iris_detector/models/iris_landmark.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        center_y = int(multi_face_landmarks[0][386][0] + (multi_face_landmarks[0][374][0] - multi_face_landmarks[0][386][0]) / 2)
        center_x = int(multi_face_landmarks[0][362][1] + (multi_face_landmarks[0][263][1] - multi_face_landmarks[0][362][1]) / 2)
        centerLeft = [center_y, center_x]
        center_y = int(multi_face_landmarks[0][159][0] + (multi_face_landmarks[0][145][0] - multi_face_landmarks[0][159][0]) / 2)
        center_x = int(multi_face_landmarks[0][130][1] + (multi_face_landmarks[0][133][1] - multi_face_landmarks[0][130][1]) / 2)
        centerRight = [center_y, center_x]

        x_scale = []
        eye_yaw = []
        y_scale = []
        eye_pitch = []
        for center, side in [[centerLeft, "left"], [centerRight, "right"]]:

            # TODO: automatically side
            orig_img = centerCropSquare(image, center,
                                   side=100)  # 400 is 1200 (image size) * 64/192, as the detector takes a 64x64 box inside the 192 image
            #if side == "right":
            #    img = np.fliplr(orig_img)  # the detector is trained on the left eye only, hence the flip

            img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))
            input_data = np.expand_dims(img.astype(np.float32) / 127.5 - 1.0, axis=0)
            input_shape = input_details[0]['shape']
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data_0 = interpreter.get_tensor(output_details[0]['index'])
            eyes = output_data_0
            iris = interpreter.get_tensor(output_details[1]["index"])
            x, y = eyes[0, ::3][0:18], eyes[0, 1::3][0:18]
            x = x / img.shape[1] * orig_image.shape[1] - 50 + center[1]
            y = y / img.shape[0] * orig_image.shape[0] - 50 + center[0]
            x_min = np.min(x)
            x_max = np.max(x)
            y_min = np.min(y)
            y_max = np.max(y)
            c_x = x_min + (x_max - x_min) / 2
            c_y = y_min + (y_max - y_min) / 2
            x_iris, y_iris = iris[0, ::3], iris[0, 1::3]

            x_iris = x_iris / img.shape[1] * orig_image.shape[1] - 50 + center[1]
            y_iris = y_iris / img.shape[0] * orig_image.shape[0] - 50 + center[0]
            for i in range(len(x_iris)):
                cv2.circle(image, (x_iris[i], y_iris[i]),
                                      radius=1, color=(255, 0, 0), thickness=1)
            i = 0
            x_s = (x_iris[0] - x_min) / (x_max - x_min)
            y_s = (y_iris[0] - y_min) / (y_max - y_min)
            x_scale.append(x_s)
            y_scale.append(y_s)
            if x_s <= 1 and x_s >= 0 and y_s <= 1 and y_s >= 0:
                eye_pitch.append(55 * x_s - 27.5)
                eye_yaw.append(70 * x_s - 35)
        """if x_scale[0] >= 0 and x_scale[0] <= 0.35 and y_scale[0] >= 0 and y_scale[0] <= 1 or x_scale[1] >= 0 and x_scale[1] <= 0.35  \
                and y_scale[1] >= 0 and y_scale[1] <= 1 :
            gaze = "right"
        elif x_scale[0] <= 1 and x_scale[0] >= 0.65  and y_scale[0] >= 0 and y_scale[0] <= 1  or x_scale[1] <= 1 and x_scale[1] >= 0.65 \
                and y_scale[1] >= 0 and y_scale[1] <= 1:
            gaze = "left"
        else:
            gaze = "center"

        if y_scale[0] >= 0 and y_scale[0] <= 0.35  and x_scale[0] >= 0 and x_scale[0] <= 1  or y_scale[1] >= 0 and y_scale[1] <= 0.35\
                and x_scale[1] >= 0 and x_scale[1] <= 1:
            gaze += " up"
        elif y_scale[0] <= 1 and y_scale[0] >= 0.65  and x_scale[0] >= 0 and x_scale[0] <= 1  or y_scale[1] <= 1 and y_scale[1] >= 0.65 and x_scale[1] >= 0 and x_scale[1] <= 1:
            gaze += " down"""""
