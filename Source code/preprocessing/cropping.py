import cv2
import numpy as np

def get_sum(img):
    # get sum of pixel on X and Y axis
    x_sum = np.sum(img, axis=0) / img.shape[1]
    y_sum = np.sum(img, axis=1) / img.shape[0]
    return x_sum, y_sum

def get_variation(array):
    # get spikes variations
    bounds = []
    b1 = 0
    b2 = 0
    flag = True
    for i, pixel in enumerate(array):
        if pixel > 4:
            if flag:
                b1 = i
                flag = False
        if pixel < 5:
            if not flag:
                b2 = i-1
                bounds.append((b1, b2))
                flag = True
    if b2 == 0 or b1 != bounds[-1][0]:
        bounds.append((b1, len(array)-1))
    return bounds

def is_close_gap(tuple1, tuple2, treshold=100):
    # Check if gap between two spikes is small
    if tuple2[0] - tuple1[1] < treshold:
        return True
    else:
        return False

def get_boundaries(tuple_array):
    # Remove small gap between spikes
    tmp = tuple_array[0]
    bound = []
    for i in range(len(tuple_array)):
        if i+1 == len(tuple_array):
            break
        if is_close_gap(tuple_array[i], tuple_array[i+1]):
            tmp = (tmp[0], tuple_array[i+1][1])
        else:
            bound.append(tmp)
            tmp = tuple_array[i+1]
    bound.append(tmp)
    return bound

def get_biggest_boundary(tuple_array):
    # Keep only the largest boundary
    diff = tuple_array[0][1] - tuple_array[0][0]
    tmp = tuple_array[0]
    for item in tuple_array:
        if diff < item[1] - item[0]:
            tmp = item
            diff = item[1] - item[0]
    return tmp

def image_cropping(img, padding=0):
    # binarize image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_inv = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Get sum of imgae by axis
    x_sum, y_sum = get_sum(img_inv)

    # Get segments with spikes by axis
    x_var = get_variation(x_sum)
    y_var = get_variation(y_sum)

    # Remove small gap between spikes
    x_bound = get_boundaries(x_var)
    y_bound = get_boundaries(y_var)

    # Keep the largest spike
    x_bound_final = get_biggest_boundary(x_bound)
    y_bound_final = get_biggest_boundary(y_bound)

    # Draw box around original image
    #cv2.rectangle(img, (x_bound_final[0] - padding, y_bound_final[0] - padding), (x_bound_final[1] + padding, y_bound_final[1] + padding), (0,255,0), 1)
    return img[y_bound_final[0]:y_bound_final[1], x_bound_final[0]:x_bound_final[1]]

