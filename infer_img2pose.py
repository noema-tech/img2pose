from collections import defaultdict
from multiprocessing.pool import ThreadPool
from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from img2pose import img2poseModel
from model_loader import load_model


transform = transforms.Compose([transforms.ToTensor()])


def main(args):
    img2pose_model = get_img2pose()

    batch_size = args.batch_size
    image_path = args.image_path

    path_batches, total_images = get_path_batches(batch_size, image_path)

    if args.undistort:
        mtx = np.load("DS-2CD2463G0_mtx.npy")
        dist = np.load("DS-2CD2463G0_dist.npy")
        input_image_shape = (3072, 1728)
        newcameramtx, (UD_X, UD_Y, UD_W, UD_H) = cv2.getOptimalNewCameraMatrix(mtx, dist, input_image_shape, 1,
                                                                               input_image_shape)
        undistort_args = (mtx, dist, newcameramtx, UD_X, UD_Y, UD_W, UD_H)
    else:
        undistort_args = None

    ret_rows = []
    pbar = tqdm(leave=False, total=total_images)
    for (person, pose, point), batches in path_batches.items():
        pbar.set_description(f"{person=} {pose=} {point=}")
        for batch in batches:
            with ThreadPool(batch_size) as pool:
                image_t = pool.map(lambda x: pre_process_image(x, undistort_args), batch)
            batch_res = img2pose_model.predict(image_t)

            for res in batch_res:
                all_bboxes = res["boxes"].cpu().numpy().astype('float')
                for i in range(len(all_bboxes)):
                    score = res["scores"][i]
                    pose_pred = res["dofs"].cpu().numpy()[i].astype('float')
                    pose_pred = pose_pred.squeeze()
                    ret_rows.append([person, pose, point] + list(pose_pred) + [score])

            pbar.update(len(batch_res))

    df = pd.DataFrame(ret_rows, columns=["person", "pose", "point", "pitch", "yaw", "roll", "x", "y", "z", "score"])
    csv_outfile = "infer_image2pose_undistort.csv" if args.undistort else "infer_image2pose.csv"
    df.to_csv(csv_outfile, index=False)


def pre_process_image(path, undistort_args):
    img = cv2.imread(str(path))
    if undistort_args:
        mtx, dist, newcameramtx, UD_X, UD_Y, UD_W, UD_H = undistort_args
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        img = dst[UD_Y:UD_Y + UD_H, UD_X:UD_X + UD_W]
    t = transform(img)
    return t


def get_path_batches(batch_size, image_path):
    total_num = 0
    paths = defaultdict(list)
    for p in image_path.glob(f"**/*.jpeg"):
        person = p.parent.parent.parent.name
        pose = p.parent.parent.name
        point = p.parent.name
        paths[(person, pose, point)].append(p)
        total_num += 1
    path_batches = {}
    for k, ps in paths.items():
        batch_n = int(np.ceil(len(ps) / batch_size))
        path_batches[k] = [ps[i * batch_size:(i + 1) * batch_size] for i in range(batch_n)]
    return path_batches, total_num


def get_img2pose(image_trace_hw=None):
    DEPTH = 18
    MAX_SIZE = 3072
    MIN_SIZE = 600
    POSE_MEAN = "models/WIDER_train_pose_mean_v1.npy"
    POSE_STDDEV = "models/WIDER_train_pose_stddev_v1.npy"
    MODEL_PATH = "models/img2pose_v1.pth"
    threed_points = np.load('pose_references/reference_3d_68_points_trans.npy')
    pose_mean = np.load(POSE_MEAN)
    pose_stddev = np.load(POSE_STDDEV)
    img2pose_model = img2poseModel(
        DEPTH, MIN_SIZE, MAX_SIZE,
        pose_mean=pose_mean, pose_stddev=pose_stddev,
        threed_68_points=threed_points,
    )
    load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
    img2pose_model.evaluate()
    if image_trace_hw:
        input_example = torch.rand(1, 3, image_trace_hw[0], image_trace_hw[1], device=img2pose_model.device)
        print(f"Trace model using {input_example.shape} tensor")
        img2pose_model.fpn_model = torch.jit.trace(img2pose_model.fpn_model, input_example)
    return img2pose_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=lambda x: Path(x))
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--undistort", action="store_true")
    main(parser.parse_args())
