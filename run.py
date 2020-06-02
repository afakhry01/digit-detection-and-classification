import os
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

PATH_REC = "./cnn/digit_classification_cnn.pt"


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 60, 5, 1)
        self.conv2 = nn.Conv2d(60, 150, 5, 1)
        self.fc1 = nn.Linear(5*5*150, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*150)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class REGION:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        self.label = 0
        self.score = 0

    def area(self):
        (x1, y1), (x2, y2) = self.pt1, self.pt2
        return (y2 - y1) * (x2 - x1)


def non_maximum_suppression(boxes, min_score):
    # 1. Set max iou value
    iou_max = 0.1

    # 2. Handle MSER return None
    if boxes is None or len(boxes) == 0:
        return []

    suppress = [False] * len(boxes)
    boxes.sort(key=lambda x: x.score, reverse=True)
    for i in range(0, len(boxes)):
        if suppress[i]:
            continue
        upper_bb = boxes[i]
        # 3. First suppression layer
        if upper_bb.score < min_score:
            suppress[i] = True
            continue
        (ux1, uy1), (ux2, uy2) = upper_bb.pt1, upper_bb.pt2
        # 4. Second suppression layer
        for j in range(i + 1, len(boxes)):
            lower_bb = boxes[j]
            (lx1, ly1), (lx2, ly2) = lower_bb.pt1, lower_bb.pt2
            lxi = max(ux1, lx1)
            lyi = max(uy1, ly1)
            uxi = min(ux2, lx2)
            uyi = min(uy2, ly2)
            intersection_area = max(0, (uyi - lyi)) * max(0, (uxi - lxi))
            iou = intersection_area / (upper_bb.area() + lower_bb.area() - intersection_area + 1e-9)
            if iou > iou_max:
                suppress[j] = True

    final_boxes = []
    for box, sup in zip(boxes, suppress):
        if not sup:
            final_boxes.append(box)

    return final_boxes


def load_LeNet(path):
    model = LeNet()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


def detect_numbers_mser(image):
    regs = []
    mser = cv2.MSER_create(_min_area=500, _max_variation=0.05, _edge_blur_size=2)

    image = cv2.medianBlur(image, 5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    regions, boxes = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    for cnt in hulls:
        x, y, w, h = cv2.boundingRect(cnt)
        reg = REGION((x, y), (x + w, y + h))
        regs.append(reg)

    return regs


def draw_boxes_and_labels(image, regions):
    for region in regions:
        cv2.rectangle(image, region.pt1, region.pt2, (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(region.label), (region.pt2[0]-23, region.pt2[1]+23), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)


def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def video_frame_generator(filename):
    # Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None


def process_video(video, fps, model, transform):

    image_gen = video_frame_generator(video)

    image = image_gen.__next__()
    h, w, d = image.shape

    out_path = "output/video.mp4"
    video_out = mp4_video_writer(out_path, (w, h), fps)

    frame_num = 1

    while image is not None:

        print("Processing fame {}".format(frame_num))

        # Begin process frame
        regions = detect_numbers_mser(image)
        new_regions = []

        for region in regions:
            (x1, y1), (x2, y2) = region.pt1, region.pt2
            img_trans = transform(Image.fromarray(image[y1:y2, x1:x2]))
            img_tensor = img_trans.unsqueeze(0)
            output = model(img_tensor)
            score, pred = torch.max(output, 1)
            region.score = score
            region.label = pred.item()
            new_regions.append(region)

        boxes = non_maximum_suppression(new_regions, 5)

        draw_boxes_and_labels(image, boxes)
        # End

        video_out.write(image)

        image = image_gen.__next__()

        frame_num += 1

    video_out.release()


def process_images_from_folder(folder, model, transform):
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is None:
            continue
        # Begin process image
        print("Processing image {}".format(filename))
        regions = detect_numbers_mser(image)
        new_regions = []

        for region in regions:
            (x1, y1), (x2, y2) = region.pt1, region.pt2
            img_trans = transform(Image.fromarray(image[y1:y2, x1:x2]))
            img_tensor = img_trans.unsqueeze(0)
            output = model(img_tensor)
            score, pred = torch.max(output, 1)
            region.score = score
            region.label = pred.item()
            new_regions.append(region)

        boxes = non_maximum_suppression(new_regions, 6)

        draw_boxes_and_labels(image, boxes)

        out_path = "output/{}".format(filename)
        cv2.imwrite(out_path, image)
        # End


def main():
    parser = argparse.ArgumentParser(description='Please add --i for images, --v for video or --all for both')
    parser.add_argument('--i', action='store_true', help='Process images 1 to 5')
    parser.add_argument('--v', action='store_true', help='Process video')
    parser.add_argument('--all', action='store_true', help='Process both images and videos')
    args = parser.parse_args()

    if not args.i and not args.v and not args.all:
        print("Please add --i for images, --v for video or --all for both")
    else:
        model = load_LeNet(PATH_REC)
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])
        if args.i or args.all:
            process_images_from_folder("input", model, transform)

        if args.v or args.all:
            video_path = "input/video.mp4"
            process_video(video_path, 29.97, model, transform)


if __name__ == '__main__':
    main()