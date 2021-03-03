import cv2
import torch
import numpy as np
import random
import pathlib
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageDraw, ImageFont
import albumentations as albu
from albumentations.pytorch import ToTensor

BORDER_CONSTANT = 0
BORDER_REFLECT = 2


class Number_Generator:
    def __init__(self, fonts_dir=r'Z:\WORK2\Numbers_generator\fonts'):
        self.true_alphabet = '0123456789'
        self.trash_alphabet = '+-=#*%$№@'
        self.alphabet = self.true_alphabet + self.trash_alphabet
        self.true_prob = 1 / 11
        self.trash_prob = 1 / 11 / len(self.trash_alphabet)
        self.probs = [self.true_prob for _ in range(len(self.true_alphabet))] + \
                     [self.trash_prob for _ in range(len(self.trash_alphabet))]
        self.indexes = np.arange(len(self.alphabet))
        self.Fonts = self._get_fonts(fonts_dir)

    def _get_fonts(self, fonts_dir):
        py = pathlib.Path(fonts_dir).glob("*.ttf")
        Fonts = []
        for file in py:
            Fonts.append(ImageFont.truetype(str(file), 64))
        return Fonts

    def Get_item(self,
                 char_to_crop_size=0.5,
                 prob_to_add_dots=0.2):

        ind = np.random.choice(self.indexes, 1, p=self.probs)[0]
        char_to_draw = self.alphabet[ind]
        true_label = char_to_draw
        ####### SIZE CHOICE ########

        font = np.random.choice(self.Fonts)
        draw_size = font.getsize(char_to_draw)
        actual_size = font.getmask(char_to_draw).size

        if actual_size[0] > actual_size[1]:  # x > y
            _max = actual_size[0]
        else:
            _max = actual_size[1]
        image_size = int(_max / char_to_crop_size * 2 - _max)
        if image_size < 100:
            image_size = 128

        ####### Coordinates to draw ######
        # drawing adds some pixels from up and 0 or 1 from down
        y_bias = draw_size[1] - actual_size[1]

        x_min = int((image_size - actual_size[0]) / 2)
        y_min = int((image_size - actual_size[1]) / 2) - y_bias

        ##### COLOR CHOICE #####
        R, G, B = np.random.randint(255, size=(3))
        R_char = np.random.choice(np.delete(np.arange(255), np.s_[R - 50:R + 50]))
        G_char = np.random.choice(np.delete(np.arange(255), np.s_[G - 50:G + 50]))
        B_char = np.random.choice(np.delete(np.arange(255), np.s_[B - 50:B + 50]))
        background_color = (R, G, B)
        text_color = (R_char, G_char, B_char)

        image = Image.new('RGB', (image_size, image_size), color=background_color)

        ###### ADD random dots to char ####
        if np.random.rand() < prob_to_add_dots:
            adding = random.choice('.,:\'')
            char_to_draw = adding + char_to_draw
        if np.random.rand() < prob_to_add_dots:
            adding = random.choice('.,:\'')
            char_to_draw = char_to_draw + adding

        draw = ImageDraw.Draw(image)
        draw.text((x_min, y_min), char_to_draw, fill=text_color, font=font, stroke_width=0)

        # BBOX [x_min, y_min, width, height] format = 'pascal_voc'
        label = int(true_label) if true_label in self.true_alphabet else 10
        bbox = [x_min, y_min + y_bias, actual_size[0] + x_min, actual_size[1] + y_min + y_bias, label]

        assert image.size[0] > 64, f'{font.path}'
        return image, bbox




import albumentations as albu
from albumentations.pytorch import ToTensor

BORDER_CONSTANT = 0
BORDER_REFLECT = 2

def pre_transforms(image_size=64):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    return albu.Compose([
        albu.CenterCrop(image_size+20,image_size+20),
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=BORDER_CONSTANT, value=0)
    ], bbox_params=albu.BboxParams(format='pascal_voc'))


def hard_transforms():
    return albu.Compose([
        albu.Rotate(limit=30, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
        albu.RandomSizedBBoxSafeCrop(width=64, height=64, erosion_rate=0.2),
        albu.InvertImg(p=0.3),
        albu.HueSaturationValue(p=0.3),
        albu.OneOf([
            albu.IAAAdditiveGaussianNoise(),
            albu.GaussNoise(),
            albu.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1)
        ], p=0.3),
        albu.OneOf([
            albu.MotionBlur(p=0.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
        ], p=0.3),
        albu.JpegCompression(quality_lower=30, quality_upper=100, p=0.5),
        albu.Cutout(num_holes=10, max_h_size=5, max_w_size=5, fill_value=0, p=0.5),
        ], p=1, bbox_params=albu.BboxParams(format='pascal_voc'))

def post_transforms():
    return albu.Compose([
        albu.Normalize(),
        ToTensor()],
        p=1, bbox_params=albu.BboxParams(format='pascal_voc'))

train_transforms = albu.Compose([
    hard_transforms(),
    post_transforms()
])

train_transforms = albu.Compose([
    hard_transforms(),
    post_transforms()
])

valid_transforms = albu.Compose([pre_transforms(), post_transforms()])

show_transforms = albu.Compose([pre_transforms(), hard_transforms()])


class Dataset_one_PIL_digit(Dataset):
    def __init__(self, transforms=None, size_of_dataset=1000) -> None:
        self.transforms = transforms
        self.size_of_dataset = size_of_dataset
        self.Generator = Number_Generator()

    def __len__(self) -> int:
        return self.size_of_dataset

    def __getitem__(self, idx: int) -> dict:
        img, bbox = self.Generator.Get_item(char_to_crop_size=0.5)

        result = {"image": np.array(img), "bboxes": [bbox]}

        if self.transforms is not None:
            result = self.transforms(**result)

        result['label'] = bbox[-1]

        return result['image'], result['label']






