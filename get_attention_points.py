import xlsxwriter
import numpy as np
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

threshold = 0.75

labels = {
    1: {
        'x': 1.5,
        'y': 1.44
    },
    2: {
        'x': 3,
        'y': 0
    },
    3: {
        'x': 0,
        'y': 0
    },
    4: {
        'x': 0,
        'y': 2
    },
    5: {
        'x': 3,
        'y': 2
    }
}

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 1024)
x_arr = []
y_arr = []
z_arr = []

camera_x = 1.5
#for low camera
#camera_y = 1.44
#for high camera
camera_y = 0.14

result_yaw = {}
result_yaw_gt = {}
result_pitch = {}
result_pitch_gt = {}

labels = {
    1: [1.5, 1.44],
    2: [3, 0],
    3: [0, 0],
    4: [0, 2],
    5: [3, 2]
}
workbook = xlsxwriter.Workbook('result_high.xlsx')

attention_y_arr = []
image_intrinsics = np.array([[1897, 0, 1536], [0, 1897, 864], [0, 0, 1]])
data_dict = {}
folder = "../digital-signage-interactive/frames_high/"
for name in os.listdir(folder):
    results = {}

    for position in os.listdir("{}{}".format(folder, name)):
        dict_x = {}
        dict_y = {}
        for point in os.listdir("{}{}/{}".format(folder, name, position)):
            print("{}{}/{}/{}".format(folder, name, position, point))
            for file in os.listdir("{}{}/{}/{}".format(folder, name, position, point)):
                img = cv2.imread("{}{}/{}/{}/".format(folder, name, position, point) + file)
                if position != '1_3' and position != '3_3':
                    img[:,2600:,:] = 0
                h, w, ch = img.shape
                #res = img2pose_model.predict([transform(cv2.resize(img, (800, 900)))])[0]
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
                        #yaw_gt = ((2.17 * yaw_gt * 180 / np.pi) - 0.43) * np.pi / 180
                        #pitch_gt = ((2.19 * pitch_gt * 180 / np.pi) - 9.6) * np.pi / 180
                        poses.append(pose_pred)

                        x, y, z = pose_pred[3:] / image_intrinsics[0, 0] * 100
                        omega = np.arctan(x/ z)
                        attention_x = camera_x + x - z * np.tan(omega + yaw_gt)
                        omega = np.arctan(y / z)  # * 180 / np.pi
                        attention_y = camera_y + y + z * np.tan(pitch_gt - omega)
                        if int(point) in dict_x.keys():
                            dict_x[int(point)].extend([attention_x])
                        else:
                            dict_x[int(point)] = [attention_x]
                        if int(point) in dict_y.keys():
                            dict_y[int(point)].extend([attention_y])
                        else:
                            dict_y[int(point)] = [attention_y]


        row = []
        for i in np.arange(1, 6, 1):
            mae_x = np.mean(abs(np.asarray(dict_x[i]) - labels[i][0]), axis=0)
            mse_x = np.mean(np.square(np.asarray(dict_x[i]) - labels[i][0]), axis=0)
            mae_y = np.mean(abs(np.asarray(dict_y[i]) - labels[i][1]), axis=0)
            mse_y = np.mean(np.square(np.asarray(dict_y[i]) - labels[i][1]), axis=0)
            dist = np.mean(np.sqrt(np.square(np.asarray(dict_x[i]) - labels[i][0]) + np.square(np.asarray(dict_y[i]) - labels[i][1])))
            print(f"Point: {i} mae_x: {mae_x:.3f} mse_x: {mse_x:.3f} mae_y: {mae_y:.3f} mse_y: {mse_y:.3f}")
            row.extend([mae_x, mae_y, dist, dist / int(position.split("_")[0]) * 100])
        results[position] = row


    worksheet = workbook.add_worksheet(name)
    row = 0
    col = 0
    merge_format = workbook.add_format({
        'bold': True,
        'align': 'center'
    })

    cell_format = workbook.add_format()
    cell_format.set_bold()

    worksheet.write(row, col, "")
    worksheet.merge_range('B1:E1', '1', merge_format)
    worksheet.merge_range('F1:I1', '2', merge_format)
    worksheet.merge_range('J1:M1', '3', merge_format)
    worksheet.merge_range('N1:Q1', '4', merge_format)
    worksheet.merge_range('R1:U1', '5', merge_format)
    row += 1

    worksheet.write(row, col, "Position", cell_format)
    for i in range(1, 21, 4):
        worksheet.write(row, col + i, "x: mae", cell_format)
        worksheet.write(row, col + i + 1, "y: mae", cell_format)
        worksheet.write(row, col + i + 2, "dist", cell_format)
        worksheet.write(row, col + i + 3, "dist / z * 100", cell_format)
    row += 1
    for item in ['1_1', '1_2', '1_3', '3_1', '3_2', '3_3', '5_1', '7_1', '9_1']:
        col = 0
        worksheet.write(row, col, item, cell_format)
        col += 1
        for i in range(len(results[item])):
            worksheet.write(row, col, results[item][i])
            col += 1
        row += 1
    col = 0
    worksheet.write(row, col, "All:", cell_format)
    col += 1
    for i in range(20):
        worksheet.write_formula(row, col, '=AVERAGE(' + chr(66 + i) + '3:' + chr(66 + i) + '11)')
        col += 1

workbook.close()