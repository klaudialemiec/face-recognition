import cv2
import time
from image_prepocessing import image_modifiers
from image_prepocessing import utils
from scipy.spatial import distance
from mtcnn.mtcnn import MTCNN

detector = MTCNN()


def detect_faces(img):
    result = detector.detect_faces(img)
    return result


def get_face_features(face):
    left_upper_x = face['box'][0]
    left_upper_y = face['box'][1]
    width = face['box'][2]
    heigth = face['box'][3]
    left_eye = face['keypoints']['left_eye']
    right_eye = face['keypoints']['right_eye']
    nose = face['keypoints']['nose']
    mouth_left = face['keypoints']['mouth_left']
    mouth_right = face['keypoints']['mouth_right']

    if left_upper_x < 0:
        width += left_upper_x
        left_upper_x = 0
    if left_upper_y < 0:
        heigth += left_upper_y
        left_upper_y = 0
    return left_upper_x, left_upper_y, width, heigth, left_eye, right_eye, nose, mouth_left, mouth_right


def find_central_face(img, faces):
    if len(faces) == 0:
        return None

    if len(faces) == 1:
        return faces[0]

    if len(faces) > 1:
        img_heigth, img_width, _ = img.shape
        img_center = int(img_width/2), int(img_heigth/2)

        best_face = None
        best_distance = 999999
        for face in faces:
            x, y, w, h, _, _, _, _, _ = get_face_features(face)
            box_center = int(x + w/2), int(y + h/2)
            dist = distance.euclidean(img_center, box_center)

            if dist < best_distance:
                best_distance = dist
                best_face = face
    return best_face


def mark_faces(img, faces):
    for face in faces:
        x, y, w, h, left_eye, right_eye, nose, mouth_left, mouth_right = get_face_features(face)

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.circle(img, left_eye, 2, (255, 0, 0), 2)
        cv2.circle(img, right_eye, 2, (255, 0, 0), 2)
        cv2.circle(img, nose, 2, (255, 0, 0), 2)
        cv2.circle(img, mouth_left, 2, (255, 0, 0), 2)
        cv2.circle(img, mouth_right, 2, (255, 0, 0), 2)
    return img


def align(img, face):
    _, _, _, _, left_eye, right_eye, _, _, _ = get_face_features(face)
    M = utils.get_rotation_matrix(left_eye, right_eye)
    img_height, img_width, _ = img.shape
    rotated = cv2.warpAffine(img, M, (img_width, img_height), flags=cv2.INTER_CUBIC)
    return rotated


def resize_image(img, height, width, interpolation=cv2.INTER_NEAREST):
    return cv2.resize(img, (height, width), interpolation=interpolation)


def crop_and_resize(img, face, height, width):
    x, y, w, h,  _, _, _, _, _ = get_face_features(face)
    img = img[y: y+h, x: x + w]
    return resize_image(img, height, width)


def save_to_file(file_path, image):
    cv2.imwrite(file_path, image)


def start_video():
    CAMERA_DEVICE_ID = 0
    faces = []
    video_capture = cv2.VideoCapture(CAMERA_DEVICE_ID)
    timeout = 60  # [seconds]
    timeout_start = time.time()

    while time.time() < timeout_start + timeout and video_capture.isOpened():
        ok, frame = video_capture.read()
        if not ok:
            print("Could not read frame from camera")
            break

        faces = detect_faces(frame)
        if len(faces) > 0:
            video_capture.release()
            cv2.destroyAllWindows()
    return frame, faces


def paint_detected_faces(image, faces, names):
    for face, name in zip(faces, names):
        x, y, w, h, _, _, _, _, _ = get_face_features(face)

        if name == 'Unknown':
            color = (0, 0, 255)
        else:
            color = (0, 128, 0)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(image, (x, y + h - 35), (x + w, y + h), color, cv2.FILLED)
        cv2.putText(image, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    return image


def detect_and_paint_face(image, name):
    faces = detect_faces(image)

    if len(faces) > 0:
        height, width, _ = image.shape
        if height > 2000 or width > 2000:
            image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

        elif height > 1000 or width > 1000:
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        faces = detect_faces(image)
        central_face = find_central_face(image, faces)
        result = paint_detected_faces(image, central_face, name)
        return result
    return None


def preprocess_image(image, desired_height, desired_width):
    results = []
    faces = detect_faces(image)

    if len(faces) > 0:
        central_face = find_central_face(image, faces)
        aligned = align(image, central_face)
        cropped = crop_and_resize(aligned, central_face, desired_height, desired_width)
        results.append(cropped)

        for face in faces:
            if face != central_face:
                aligned = align(image, face)
                cropped = crop_and_resize(aligned, face, desired_height, desired_width)
                results.append(cropped)
    return results


def preprocess_images(images, desired_height, desired_width):
    return [preprocess_image(img, desired_height, desired_width) for img in images]

