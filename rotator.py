import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os


def rotate_image(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    image = cv2.resize(image, (w, h))

    return image


def get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def rotate_box(corners, angle, cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0])*(bbox[:, 3] - bbox[:, 1])


def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image

    Parameters
    ----------

    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.

    Returns
    -------

    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    delta_area = ((ar_ - bbox_area(bbox)) / ar_)

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1, :]

    return bbox


def main_rotator(img, bboxes, angle):

    w, h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2

    img = rotate_image(img, angle)

    corners = get_corners(bboxes)

    corners = np.hstack((corners, bboxes[:, 4:]))

    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)

    scale_factor_x = img.shape[1] / w

    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w, h))

    new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

    bboxes = new_bbox

    bboxes = clip_box(bboxes, np.array([0, 0, w, h]), 0.25)

    return img, bboxes


directory_path = '/Users/tina/Desktop/images/tr2'

all_files = os.listdir(directory_path)
jpg_files = [file for file in all_files if file.endswith('.jpg')]


for _ in range(len(jpg_files)):
    file_name = np.random.choice(jpg_files)[:-4]

    image_path = os.path.join(directory_path, f'{file_name}.jpg')
    xml_file_path = os.path.join(directory_path, f'{file_name}.xml')

    image = cv2.imread(image_path)[:, :, ::-1]

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    bboxes_raw = []

    # Extract bounding box coordinates
    for obj in root.findall(".//object"):
        name = obj.find("name").text
        bbox_elem = obj.find(".//bndbox")
        xmin = float(bbox_elem.find("xmin").text)
        ymin = float(bbox_elem.find("ymin").text)
        xmax = float(bbox_elem.find("xmax").text)
        ymax = float(bbox_elem.find("ymax").text)

        bbox_arr = np.array([xmin, ymin, xmax, ymax])

        bboxes_raw.append(bbox_arr)

    bboxes_raw = np.array(bboxes_raw)

    angle_count = 0
    prev_angle = 0

    while angle_count < 4:
        angle = np.random.randint(-30, 30)

        if abs(angle - prev_angle) > 10:  # Not wanting rotations to be very similar to each other
            image_output, bboxes_output = main_rotator(image, bboxes_raw, angle)

            cv2.imwrite(os.path.join(directory_path, f'{file_name}_rotated{angle_count}.jpg'), image_output)

            for i, obj in enumerate(root.findall(".//object")):
                name = obj.find("name").text
                bbox_elem = obj.find(".//bndbox")

                bbox_elem.find("xmin").text = str(bboxes_output[i][0])
                bbox_elem.find("ymin").text = str(bboxes_output[i][1])
                bbox_elem.find("xmax").text = str(bboxes_output[i][2])
                bbox_elem.find("ymax").text = str(bboxes_output[i][3])
                #

                prev_angle = angle

            tree.write(os.path.join(directory_path, f'{file_name}_rotated{angle_count}.xml'))
            angle_count += 1
        else:
            continue

