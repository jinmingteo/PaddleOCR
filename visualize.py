from paddleocr import PaddleOCR, draw_ocr, str2bool
from PIL import Image, ImageDraw, ImageFont
import os 
import math
from ppstructure.utility import init_args
import numpy as np

def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    print ("HI")
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon([
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ], fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)

def parse_args():
    import argparse
    parser = init_args()
    parser.add_argument("--lang", type=str, default='japan')
    parser.add_argument("--font", type=str, default='doc/fonts/japan.ttc')
    parser.add_argument("--img", type=str, default='./ppocr_img/imgs/japan')
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--save_results", type=str2bool, default=True)
    parser.add_argument(
        "--ocr_version",
        type=str,
        default='PP-OCRv2',
        help='OCR Model version, the current model support list is as follows: '
        '1. PP-OCRv2 Support Chinese detection and recognition model. '
        '2. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.'
    )

    return parser.parse_args()
    
if __name__ == "__main__":
    params = parse_args()
    # The model file will be downloaded automatically when executed for the first time
    ocr = PaddleOCR(lang=params.lang, det=params.det, rec=params.rec)
    if os.path.isdir(params.img):
        img_path = [os.path.join(params.img, item) for item in os.listdir(params.img)]
    else:
        img_path = [params.img]
    
    for img in img_path:
        print (f"OCR for {img}")
        result = ocr.ocr(img)
        for line in result:
            print(line)
        
        if params.save_results:
            # Visualization
            base_img = os.path.basename(img)
            image = Image.open(img).convert('RGB')
            boxes = [line[0] for line in result]
            #boxes = [np.reshape(np.array(line[0]), [-1,1,2]).astype(np.int64) for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            im_show = draw_ocr_box_txt(image, boxes, txts, scores, font_path='doc/fonts/japan.ttc')
            im_show = Image.fromarray(im_show)
            os.makedirs('output', exist_ok=True)
            im_show.save(os.path.join('output', base_img))