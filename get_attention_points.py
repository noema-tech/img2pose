import sys
import xlsxwriter
import numpy as np
from torchvision import transforms
import os
from utils.renderer import Renderer
from img2pose import img2poseModel
from model_loader import load_model
import tensorflow as tf
import cv2
from mediapipe.python.solutions import face_mesh as mp_faces
np.set_printoptions(suppress=True)

def centerCropSquare(img, center, side=None, scaleWRTHeight=None):
    a = side is None
    b = scaleWRTHeight is None
    assert (not a and b) or (a and not b)  # Python doesn't have "xor"... C'mon Python!
    half = 0
    if side is None:
        half = int(img.shape[0] * scaleWRTHeight / 2)
    else:
        half = int(side / 2)

    return img[(center[0] - half):(center[0] + half), (center[1] - half):(center[1] + half), :]

def localToGlobal(img, center, side=None, scaleWRTHeight=None):
    a = side is None
    b = scaleWRTHeight is None
    assert (not a and b) or (a and not b)  # Python doesn't have "xor"... C'mon Python!
    half = 0
    if side is None:
        half = int(img.shape[0] * scaleWRTHeight / 2)
    else:
        half = int(side / 2)

    return img[(center[0] - half):(center[0] + half), (center[1] - half):(center[1] + half), :]

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

# change to a folder with images, or another list containing image paths
images_path = "../../images"

threshold = 0.75

x_arr = []
y_arr = []
z_arr = []

camera_x = 1.5
#for low camera
camera_y = 1.44
#for high camera
#camera_y = 0.14

result_yaw = {}
result_yaw_gt = {}
result_pitch = {}
result_pitch_gt = {}

labels = {
    1: {
        'x1': -0.75,
        'x2': 0.75,
        'y1': -0.5,
        'y2': 0.5
    },
    2: {
        'x1': 0.75,
        'x2': 2.25,
        'y1': -0.5,
        'y2': 0.5
    },
    3: {
        'x1': 2.25,
        'x2': 3.75,
        'y1': -0.5,
        'y2': 0.5
    },
    4: {
        'x1': -0.75,
        'x2': 0.75,
        'y1': 0.5,
        'y2': 1.5
    },
    5: {
        'x1': 0.75,
        'x2': 2.25,
        'y1': 0.5,
        'y2': 1.5
    },
    6: {
        'x1': 2.25,
        'x2': 3.75,
        'y1': 0.5,
        'y2': 1.5
    },
    7: {
        'x1': -0.75,
        'x2': 0.75,
        'y1': 1.5,
        'y2': 2.5
    },
    8: {
        'x1': 0.75,
        'x2': 2.25,
        'y1': 1.5,
        'y2': 2.5
    },
    9: {
        'x1': 2.25,
        'x2': 3.75,
        'y1': 1.5,
        'y2': 2.5
    },
}

image_intrinsics = np.array([[1897, 0, 1536], [0, 1897, 864], [0, 0, 1]])
folder = "../digital-signage-interactive/frames/"
arr_x = []
arr_y = []
for name in ['alexSh']:#os.listdir(folder):
    results = {}
    dict_x = {}
    dict_y = {}
    for position in ['1_1', '1_2', '1_3', '3_1', '3_2', '3_3', '5_1']:#os.listdir("{}{}".format(folder, name)):
        #dict_x = {}
        #dict_y = {}
        for point in ['2', '3', '4', '5']:#os.listdir("{}{}/{}".format(folder, name, position)):
            print("{}{}/{}/{}".format(folder, name, position, point))

            arr_x = []
            arr_y = []
            prob_dict = {}
            arr = []
            for file in os.listdir("{}{}/{}/{}".format(folder, name, position, point))[0:20]:
                img = cv2.imread("{}{}/{}/{}/".format(folder, name, position, point) + file)
                if position != '1_3' and position != '3_3':
                    img[:,2600:,:] = 0
                h, w, ch = img.shape
                res = img2pose_model.predict([transform(img)])[0]

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

                        faces = mp_faces.FaceMesh(
                            static_image_mode=True, min_detection_confidence=0.5)

                        if position.split('_')[0] == '1':
                            image = img[200:700, 1250:1750, :]  # 1, 3
                        if position.split('_')[0] == '3':
                            image = img[450:950, 1250:1750, :]  # 1, 3
                        if position.split('_')[0] == '5':
                            image = img[500:1000, 1200:1700, :]  # 1, 3
                        if position.split('_')[0] == '7':
                            image = img[500:1000, 1200:1700, :]  # 1, 3
                        if position.split('_')[0] == '9':
                            image = img[600:1100, 1200:1700, :]

                        face_results = faces.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        multi_face_landmarks = []
                        if face_results.multi_face_landmarks:
                            for landmarks in face_results.multi_face_landmarks:
                                x = [landmark.x for landmark in landmarks.landmark]
                                y = [landmark.y for landmark in landmarks.landmark]
                                face_landmarks = np.transpose(np.stack((y, x))) * image.shape[0:2]
                                multi_face_landmarks.append(face_landmarks)

                            faces.close()

                            interpreter = tf.lite.Interpreter(
                                model_path="../digital-signage-interactive/digital_signage_demo/analytics/iris_detector/models/iris_landmark.tflite")
                            interpreter.allocate_tensors()

                            # Get input and output tensors.
                            input_details = interpreter.get_input_details()
                            output_details = interpreter.get_output_details()

                            # Test the model on image
                            # img = cv2.imread("../test.jpg")
                            center_y = int(multi_face_landmarks[0][386][0] + (
                                    multi_face_landmarks[0][374][0] - multi_face_landmarks[0][386][0]) / 2)
                            center_x = int(multi_face_landmarks[0][362][1] + (
                                    multi_face_landmarks[0][263][1] - multi_face_landmarks[0][362][1]) / 2)
                            centerLeft = [center_y, center_x]
                            center_y = int(multi_face_landmarks[0][159][0] + (
                                    multi_face_landmarks[0][145][0] - multi_face_landmarks[0][159][0]) / 2)
                            center_x = int(multi_face_landmarks[0][130][1] + (
                                    multi_face_landmarks[0][133][1] - multi_face_landmarks[0][130][1]) / 2)
                            centerRight = [center_y, center_x]

                            x_scale = []
                            eye_yaw = []
                            y_scale = []
                            eye_pitch = []
                            for center, side in [[centerLeft, "left"], [centerRight, "right"]]:
                                side = 100
                                crop_image = centerCropSquare(image, center,
                                                            side=side)  # 400 is 1200 (image size) * 64/192, as the detector takes a 64x64 box inside the 192 image
                                # if side == "right":
                                #    img = np.fliplr(orig_img)  # the detector is trained on the left eye only, hence the flip
                                if crop_image.shape[0] > 0:

                                    img2 = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                                    img2 = cv2.resize(img2, (64, 64))
                                    input_data = np.expand_dims(img2.astype(np.float32) / 127.5 - 1.0, axis=0)
                                    input_shape = input_details[0]['shape']
                                    interpreter.set_tensor(input_details[0]['index'], input_data)
                                    interpreter.invoke()
                                    output_data_0 = interpreter.get_tensor(output_details[0]['index'])
                                    eyes = output_data_0
                                    iris = interpreter.get_tensor(output_details[1]["index"])
                                    eye_x, eye_y = eyes[0, ::3][0:16], eyes[0, 1::3][0:16]
                                    eye_x = eye_x / img2.shape[1] * crop_image.shape[1] - side / 2 + center[1]
                                    eye_y = eye_y / img2.shape[0] * crop_image.shape[0] - side / 2 + center[0]
                                    x_min = np.min(eye_x)
                                    x_max = np.max(eye_x)
                                    y_min = np.min(eye_y)
                                    y_max = np.max(eye_y)
                                    c_x = x_min + (x_max - x_min) / 2
                                    c_y = y_min + (y_max - y_min) / 2
                                    for i in range(len(eye_x)):
                                        cv2.circle(image, (eye_x[i], eye_y[i]),
                                                   radius=1, color=(0, 255, 0), thickness=1)

                                    x_iris, y_iris = iris[0, ::3], iris[0, 1::3]
                                    x_iris = x_iris / img2.shape[1] * crop_image.shape[1] - side / 2 + center[1]
                                    y_iris = y_iris / img2.shape[0] * crop_image.shape[0] - side / 2 + center[0]
                                    for i in range(len(x_iris)):
                                        cv2.circle(image, (x_iris[i], y_iris[i]),
                                                   radius=1, color=(255, 0, 0), thickness=1)
                                    i = 0
                                    x_s = (x_iris[0] - x_min) / (x_max - x_min)
                                    y_s = (y_iris[0] - y_min) / (y_max - y_min)
                                    if x_s >= 0 and x_s <= 1 and y_s >= 0 and y_s <= 1:
                                        x_scale.append(x_s)
                                        y_scale.append(y_s)
                                    if x_s <= 1 and x_s >= 0 and y_s <= 1 and y_s >= 0:
                                        eye_pitch.append(55 * y_s - 27.5)
                                        eye_yaw.append(70 * x_s - 35)
                        #if pitch_gt < 0:
                        #    pitch_gt *= 2
                        if len(eye_yaw) > 0 and len(eye_pitch) > 0:
                            yaw_gt = yaw_gt - min(eye_yaw, key=abs) * np.pi/ 180
                            pitch_gt = pitch_gt + min(eye_pitch, key=abs) * np.pi/ 180

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (10, 500)
                        fontScale = 6
                        fontColor = (255, 0, 255)
                        lineType = 4

                        cv2.putText(img, '{}, {}'.format(int(yaw_gt * 180 / np.pi), int(pitch_gt * 180 / np.pi)),
                                    (250, 250),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        x, y, z = pose_pred[3:] / image_intrinsics[0, 0] * 100

                        cv2.putText(img, '{:.2f}, {:.2f}, {:.2f}'.format(x, y, z),
                                    (250, 650),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                        omega = np.arctan(x / z)
                        attention_x = camera_x + x - z * np.tan(yaw_gt + omega)
                        omega = np.arctan(y / z)  # * 180 / np.pi
                        attention_y = camera_y + y + z * np.tan(pitch_gt - omega)
                        cv2.putText(img, '{:.2f}, {:.2f}'.format(attention_x, attention_y),
                                    (250, 850),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        #if len(eye_yaw) > 0 and len(eye_pitch) > 0:
                        #    attention_x = attention_x + np.tan(min(eye_yaw, key=abs) * np.pi / 180) * z
                       #     attention_y = attention_y + np.tan(min(eye_pitch, key=abs) * np.pi / 180) * z

                        """w2 = np.tan(15 * np.pi / 180) * z
                        h2 = np.tan(15 * np.pi / 180) * z

                        window = {
                            'x1': attention_x - w2,
                            'x2': attention_x + w2,
                            'y1': attention_y - w2,
                            'y2': attention_y + w2
                        }"""

                        """if attention_x >= labels["{}_{}".format(point, 'x')][0] and attention_x <= labels["{}_{}".format(point, 'x')][1]:
                            arr_x.append(1)
                        else:
                            arr_x.append(0)
                        if attention_y >= labels["{}_{}".format(point, 'y')][0] and attention_y <= labels["{}_{}".format(point, 'y')][1]:
                            arr_y.append(1)
                        else:
                            arr_y.append(0)"""

                        for key in labels.keys():
                            if attention_x <= labels[key]['x2'] and attention_x >= labels[key]['x1'] \
                                    and attention_y <= labels[key]['y2'] and attention_y >= labels[key]['y1']:
                                arr.append(key)


                        cv2.putText(img, '{:.2f}, {:.2f}'.format(attention_x, attention_y),
                                    (250, 1050),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                cv2.imshow('', cv2.resize(img, (1200, 800)))
                cv2.waitKey()
           
