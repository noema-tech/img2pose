import numpy as np
from torchvision import transforms
import os

from gaze_tracking.gaze_tracking import GazeTracking
from img2pose import img2poseModel
from model_loader import load_model
import cv2
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

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50):
    yaw = -yaw

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

threed_points = np.load('pose_references/reference_3d_68_points_trans.npy')
#threed_points = np.load('/app/detectors/img2pose/pose_references/reference_3d_5_points_trans.npy')
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

"""labels = {
    1: {
        'x1': 0,
        'x2': 1,
        'y1': 0,
        'y2': 0.67
    },
    2: {
        'x1': 1,
        'x2': 2,
        'y1': 0,
        'y2': 0.67
    },
    3: {
        'x1': 2,
        'x2': 3,
        'y1': 0,
        'y2': 0.67
    },
    4: {
        'x1': 0,
        'x2': 1,
        'y1': 0.67,
        'y2': 1.34
    },
    5: {
        'x1': 1,
        'x2': 2,
        'y1': 0.67,
        'y2': 1.34
    },
    6: {
        'x1': 2,
        'x2': 3,
        'y1': 0.67,
        'y2': 1.34
    },
    7: {
        'x1': 0,
        'x2': 1,
        'y1': 1.34,
        'y2': 2
    },
    8: {
        'x1': 1,
        'x2': 2,
        'y1': 1.34,
        'y2': 2
    },
    9: {
        'x1': 2,
        'x2': 3,
        'y1': 1.34,
        'y2': 2
    },
}"""

image_intrinsics = np.array([[1897, 0, 1536], [0, 1897, 864], [0, 0, 1]])
folder = "../digital-signage-interactive/frames/"
arr_x = []
arr_y = []
gaze = GazeTracking()

names_arr = []
position_arr = []
point_arr = []
iris_point_left_arr = []
center_left_arr = []
eye_center_left_arr = []
radius_left_arr = []
eye_lms_left_arr = []
iris_point_right_arr = []
center_right_arr = []
eye_center_right_arr = []
radius_right_arr = []
eye_lms_right_arr = []
detected_arr = []

for name in ['alexSh', 'alexF', 'roman']:

    results = {}
    dict_x = {}
    dict_y = {}
    for position in ['1_1', '1_2', '1_3', '3_1', '3_2', '3_3', '5_1']:

    #os.listdir("{}{}".format(folder, name)):
        #dict_x = {}
        #dict_y = {}
        #os.listdir(folder):#os.listdir("{}{}/{}".format(folder, name, position)):
        for point in ['1', '2', '3', '4', '5']:
            print("{}{}/{}/{}".format(folder, name, position, point))

            arr_x = []
            arr_y = []
            prob_dict = {}
            arr = []
            attention_x_arr = []
            attention_y_arr = []
            attention_x_with_ang_arr = []
            attention_y_with_ang_arr = []
            attention_x_with_ang_arr2 = []
            attention_y_with_ang_arr2 = []
            for file in os.listdir("{}{}/{}/{}".format(folder, name, position, point))[0:20]:
                img = cv2.imread("{}{}/{}/{}/".format(folder, name, position, point) + file)
                if position != '1_3' and position != '3_3':
                    img[:,2600:,:] = 0
                if position == '1_1':
                    frame_img = img[200:700, 1250:1750, :]  # 1, 3
                if position == '1_2':
                    frame_img = img[200:700, 50:550, :]  # 1, 3
                if position == '1_3':
                    frame_img = img[200:700, 2550:3000, :]  # 1, 3
                if position == '3_1':
                    frame_img = img[450:950, 1250:1750, :]  # 1, 3
                if position == '3_2':
                    frame_img = img[450:950, 250:750, :]  # 1, 3
                if position == '3_3':
                    frame_img = img[450:950, 2250:2750, :]  # 1, 3
                if position == '5_1':
                    frame_img = img[500:1000, 1200:1700, :]  # 1, 3
                if position == '7_1':
                    frame_img = img[500:1000, 1200:1700, :]  # 1, 3
                if position == '9_1':
                    frame_img = img[600:1100, 1200:1700, :]
                frame_img = img
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


                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (10, 500)
                        fontScale = 2
                        fontColor = (255, 0, 255)
                        lineType = 4

                        cv2.putText(img, 'head angles:{}, {}'.format(int(yaw_gt * 180 / np.pi), int(pitch_gt * 180 / np.pi)),
                                    (250, 250),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        #x, y, z = pose_pred[3:]
                        #y *= - 1
                        #x = x / 10
                        #z = z / 10
                        #y = y - 0.2
                        #z = 0.51 * (z ** 2 - x ** 2) / 10 + 0.42
                        #x = pose_pred[3] * 13 / 100
                        #y = pose_pred[4] * 15 / 100
                        #z = pose_pred[5] * 5 / 100
                        x, y = pose_pred[3:5] / 10
                        z = pose_pred[5] * image_intrinsics[0, 0] / (w + h) / 10
                        #x, y = pose_pred[3:5] / 10
                        #y = -0.4
                        #if point == '1_1' or point == '3_1' or point == '5_1':
                        #x = 0
                        #if point == '1_2':
                        #    x = 0
                        #z = int(position.split("_")[0])

                        #if z == 1:
                        #    pitch_gt *= 2
                        #    yaw2pose
                        #    _gt *= 2
                        #pose_pred[5] * image_intrinsics[0, 0] / (w + h) / 10
                        #cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                        #              (int(bbox[2]), int(bbox[3])),
                        #              (0, 0, 255), 5)
                        #cv2.rectangle(img, (500, 500), (int(500 + (h + w) / pose_pred[5]), int(500 + (h + w) / pose_pred[5])), (0, 0, 255), 5)
                        #cv2.rectangle(img, (50, 650), (350, 850), (0, 0, 255), 10)
                        cv2.putText(img, 'xyz: {:.2f}, {:.2f}, {:.2f}'.format(x, y, z),
                                    (250, 450),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        omega = np.arctan(x / z)
                        attention_x = camera_x + x - z * np.tan(yaw_gt + omega)
                        omega = np.arctan(y / z)  # * 180 / np.pi
                        attention_y = camera_y + y + z * np.tan(pitch_gt - omega)
                        attention_x_arr.append(attention_x)
                        attention_y_arr.append(attention_y)
                        cv2.putText(img, 'att_x, att_y: {:.2f}, {:.2f}'.format(attention_x, attention_y),
                                    (250, 650),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        gaze.refresh(frame_img)

                        frame = gaze.annotated_frame()
                        #cv2.imshow("", frame)
                        #cv2.waitKey()
                        img[h - frame.shape[0]:h, w - frame.shape[1]:w, :] = frame

                        yaw = yaw_gt
                        pitch = pitch_gt
                        if gaze.pupils_located:
                            yaw_gt += gaze.yaw_angle() * np.pi / 180
                            pitch_gt += gaze.pitch_angle() * np.pi / 180
                        cv2.putText(img, 'eyes angles: {}, {}'.format(int(gaze.yaw_angle()), int(gaze.pitch_angle())),
                                    (250, 850),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        omega = np.arctan(x / z)
                        attention_x = camera_x + x - z * np.tan(yaw_gt + omega)
                        omega = np.arctan(y / z)  # * 180 / np.pi
                        attention_y = camera_y + y + z * np.tan(pitch_gt - omega)
                        attention_x_with_ang_arr.append(attention_x)
                        attention_y_with_ang_arr.append(attention_y)
                        cv2.putText(img, 'att_x, att_y with eyes: {:.2f}, {:.2f}'.format(attention_x, attention_y),
                                    (250, 1050),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                        pitch, yaw, roll, _, _, scale = pose_pred
                        tdx = bbox[0] + ((bbox[2] - bbox[0]) / 2)
                        tdy = bbox[1] + ((bbox[3] - bbox[1]) / 2)

                        img = draw_axis(np.asarray(img), yaw, pitch, roll, tdx=tdx, tdy=tdy , size=1000 / scale)

                        yaw_gt = yaw
                        pitch_gt = pitch
                        if gaze.pupils_located:
                            yaw_gt += gaze.yaw_angle2() * np.pi / 180
                            pitch_gt += gaze.pitch_angle2() * np.pi / 180
                        cv2.putText(img, 'eyes angles: {}, {}'.format(int(gaze.yaw_angle2()), int(gaze.pitch_angle2())),
                                    (250, 150),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        omega = np.arctan(x / z)
                        attention_x = camera_x + x - z * np.tan(yaw_gt + omega)
                        omega = np.arctan(y / z)  # * 180 / np.pi
                        attention_y = camera_y + y + z * np.tan(pitch_gt - omega)
                        attention_x_with_ang_arr2.append(attention_x)
                        attention_y_with_ang_arr2.append(attention_y)
                        cv2.putText(img, 'att_x, att_y with eyes: {:.2f}, {:.2f}'.format(attention_x, attention_y),
                                    (250, 1450),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        """for key in labels.keys():
                            if attention_x <= labels[key]['x2'] and attention_x >= labels[key]['x1'] \
                                    and attention_y <= labels[key]['y2'] and attention_y >= labels[key]['y1']:
                                arr.append(key)
                                result = key"""


                cv2.imshow('', cv2.resize(img, (1200, 800)))
                cv2.waitKey()
            img = cv2.resize(img, (1200, 900))

            blank_image = np.zeros((300, 450, 3), np.uint8)
            cv2.circle(blank_image, (75, 50),
                       radius=10, color=(255, 0, 0), thickness=-1)
            cv2.circle(blank_image, (375, 50),
                       radius=10, color=(255, 0, 0), thickness=-1)
            cv2.circle(blank_image, (75, 250),
                       radius=10, color=(255, 0, 0), thickness=-1)
            cv2.circle(blank_image, (375, 250),
                       radius=10, color=(255, 0, 0), thickness=-1)
            cv2.circle(blank_image, (225, 194),
                       radius=10, color=(255, 0, 0), thickness=-1)
            for i in range(len(attention_x_arr)):
                blank_image = cv2.circle(blank_image, (int(attention_x_arr[i] * 100 + 75),int(attention_y_arr[i] * 100) + 50),
                                         radius=2, color=(0, 0, 255), thickness=-1)
            for i in range(len(attention_x_with_ang_arr)):
                blank_image = cv2.circle(blank_image, (int(attention_x_with_ang_arr[i] * 100 + 75),
                                                       int(attention_y_with_ang_arr[i] * 100) + 50), radius=2, color=(0, 255, 0), thickness=-1)
            for i in range(len(attention_x_with_ang_arr2)):
                blank_image = cv2.circle(blank_image, (int(attention_x_with_ang_arr2[i] * 100 + 75),
                                                       int(attention_y_with_ang_arr2[i] * 100) + 50), radius=2, color=(255, 0, 0), thickness=-1)
            cv2.imwrite("results/images4/{}_{}_{}.jpg".format(name, position, point), blank_image)

