# -*- coding: utf-8 -*-
import logging
import random

import cv2
from torchvision import transforms

from utils.models import *
from utils import utils
from utils.progress_bar import create_progressbar

DEBUG = True


def load_model(cfg_path, weights, class_names, img_size):
    if DEBUG:
        logging.info('loading darknet weights.')
    model = Darknet(cfg_path, img_size=img_size)
    model.load_weights(weights)
    model.cuda()
    model.eval()
    classes = utils.load_classes(class_names)
    return model, classes


def detect_image(img, model):
    Tensor = torch.cuda.FloatTensor

    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh - imw) / 2), 0),
                                                         max(int((imw - imh) / 2), 0), max(int((imh - imw) / 2), 0),
                                                         max(int((imw - imh) / 2), 0)), (128, 128, 128)),
                                         transforms.ToTensor(),
                                         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80,
                                               conf_thres, nms_thres)
    return detections[0]


def load_image(path):
    return Image.open(path)


def load_video(path) -> list:
    if DEBUG:
        logging.info(f'loading video from {path}')
    video = cv2.VideoCapture(path)
    success, image = video.read()
    frames = []
    while success:
        frames.append(image)  # save frame as JPEG file
        success, image = video.read()
    video.release()
    return frames


def detect_from_video(video, callback, darknet_model, color_list, fps=24, display_video=True, save_final_video=False,
                      save_path=None):
    if save_final_video and save_path is None:
        logging.info('\'save_final_video\' is set to true but \'save_path\' is empty. Please give'
                     'an output path!')
        exit(-1)
    video_frames = load_video(video)
    video_out = None
    vw, vh = video_frames[0].shape[0], video_frames[0].shape[1]
    if save_final_video:
        # xvid = cv2.VideoWriter_fourcc(*'XVID')
        mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(save_path, mp4, fps, (vh, vw))
    logging.info(f'{len(video_frames)} frames are loaded!')
    if not save_final_video and not display_video:
        logging.info("save_final_video and display_video keys are set to false! What is the point?")
        exit(-1)
    bar = create_progressbar(len(video_frames))
    for frame in video_frames:
        processed_frame = callback(frame, darknet_model, display_video, color_list)
        if save_final_video:
            video_out.write(processed_frame)
        bar.next()
    bar.finish()
    if save_final_video:
        video_out.release()


def detection_callback(frame, model, display, color_list):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = np.array(frame)
    pil_image = Image.fromarray(img)

    detections = detect_image(pil_image, model)

    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:

        unique_labels = detections[:, -1].cpu().unique()
        # n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = color_list[int(cls_pred)]
            cls = classes[int(cls_pred)]
            info = f'{cls} - {int(cls_pred)} - Coord: {int(x1)},{int(y1)},{int(x2)},{int(y2)}'
            cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 2)
            # cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 80, y1), color, -1)
            cv2.putText(frame, info, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, fontScale=1.5,
                        color=color, thickness=1)
    # frame = cv2.resize(frame, (1024, 1024))
    if display:
        cv2.imshow('Stream', frame)
        cv2.waitKey(1)
    return frame
    # outvideo.write(frame)


if __name__ == '__main__':
    import argparse as p

    parser = p.ArgumentParser(
        allow_abbrev=False,
        description='Object Detection program that implements DarkNet.'
    )
    parser.add_argument('video', help='Input video.')

    parser.add_argument('configuration', help='Configuration file path. Ends with .cfg '
                                              'extension')
    parser.add_argument('weights', help='Weights file that is downloaded from Darknet. Can be openimages'
                                        ', coco or a custom dataset. Ends with .weights.')
    parser.add_argument('names', help='Class names over rows, specified in a .names file.')
    parser.add_argument('-ct', dest='cf', type=float, default=0.9,
                        help='Confidence threshold of the model. Default value is 0.9')
    parser.add_argument('-nms', dest='nms', type=float, default=0.4,
                        help='Non-maximum suppression threshold of the model. Default value is 0.4')
    parser.add_argument('-fps', dest='fps', type=int, default=24, help='FPS of the input video. Default value is 24.')
    parser.add_argument('--d', dest='display', default=True, action='store_false',
                        help='Displays the processed video if set to true. Default is True')
    parser.add_argument('--s', dest='save', default=False, action='store_true',
                        help='Specifies if the processed video should be saved. Default is False')
    parser.add_argument('-path', dest='path', help='Specifies the processed video save path.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    img_size = 416
    darknet, classes = load_model(cfg_path=args.configuration,
                                  weights=args.weights,
                                  class_names=args.names,
                                  img_size=img_size,
                                  )
    conf_thres, nms_thres = args.cf, args.nms
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(classes) + 1)]
    detect_from_video(video=args.video,
                      callback=detection_callback,
                      color_list=colors,
                      darknet_model=darknet,
                      fps=args.fps,
                      display_video=args.display,
                      save_final_video=args.save,
                      save_path=args.path)
