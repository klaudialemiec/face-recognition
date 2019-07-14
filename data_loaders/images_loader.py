import glob
import os
import cv2
from image_prepocessing.preprocessing import preprocess_image

DEFAULT_PATH = "../data/lfw/"


def load_images(images_path=DEFAULT_PATH, min_person_images_number=1):
    persons_images_over_limit = {}
    persons_images_under_limit = {}
    persons_with_images = os.listdir(images_path)
    for person in persons_with_images:
        person_images_files = glob.glob(images_path + person + '\\*.jpg')
        if len(person_images_files) >= min_person_images_number:
            images = [cv2.imread(file) for file in person_images_files]
            persons_images_over_limit[person] = images
        else:
            images = [cv2.imread(file) for file in person_images_files]
            persons_images_under_limit[person] = images

    return persons_images_over_limit, persons_images_under_limit


def load_images_as_list(path=DEFAULT_PATH, min_person_images_number=2):
    names_images_dict, _ = load_images(path, min_person_images_number)
    names = []
    images = []
    for user, image in names_images_dict.items():
        images.extend(image)
        names.extend([user] * len(image))
    return names, images


def load_images_for_person(person, images_path=DEFAULT_PATH):
    person_images_files = glob.glob(images_path + person + '/*.jpg')
    images = [cv2.imread(file) for file in person_images_files]
    return images
