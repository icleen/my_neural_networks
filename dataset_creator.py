import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import json
import codecs


if __name__ == "__main__":
    text_lines_path = sys.argv[1]
    output_path = sys.argv[2]
    number_of_examples = sys.argv[3]
    number_of_examples = int(number_of_examples)
    try:
        os.makedirs(output_path)
    except:
        pass

    with codecs.open(text_lines_path, encoding="utf8") as f:
        text_lines = f.readlines()

    text_lines = [t.strip() for t in text_lines]
    text_lines = [t for t in text_lines if len(t) > 0]
    # print text_lines

    all_fonts = []
    for root, dirs, files in os.walk("fonts2"):
        for file in files:
            if file.endswith(".ttf"):
                 all_fonts.append(os.path.join(root, file))
            elif file.endswith(".TTF"):
                  all_fonts.append(os.path.join(root, file))


    train_data =[]
    train_cnt = int(number_of_examples * 0.9)
    val_data =[]
    val_cnt = number_of_examples - train_cnt

    font_index = 0
    for num in range(train_cnt):
        if num % 45 == 0:
            font_index += 1
            font_index %= len(all_fonts) - 1
        t = text_lines[random.randint(0, len(text_lines) - 1)]
        f = ImageFont.truetype(all_fonts[font_index], random.randint(18, 22))
        img = Image.new("RGB", (32, 32), (255,255,255))
        drawing = ImageDraw.Draw(img)
        drawing.text((random.randint(0, 5), random.randint(0, 5)), t, (0,0,0), font=f)

        open_cv_image = np.array(img)

        cv2.imwrite(output_path + '/test_' + str(num) + '.png', open_cv_image)
        train_data.append({
            "image_path": '/test_' + str(num) + '.png',
            "gt": t
        })

    font_index = len(all_fonts) - 1
    for num in range(val_cnt):
        t = text_lines[random.randint(0, len(text_lines) - 1)]
        f = ImageFont.truetype(all_fonts[font_index], random.randint(18, 22))
        img = Image.new("RGB", (32, 32), (255,255,255))
        drawing = ImageDraw.Draw(img)
        drawing.text((random.randint(0, 5), random.randint(0, 5)), t, (0,0,0), font=f)

        open_cv_image = np.array(img)

        cv2.imwrite(output_path + '/test_' + str(num + train_cnt) + '.png', open_cv_image)
        val_data.append({
            "image_path": '/test_' + str(num + train_cnt) + '.png',
            "gt": t
        })

    with open("training.json", 'w') as f:
        json.dump(train_data, f)

    with open("validation.json", 'w') as f:
        json.dump(val_data, f)
