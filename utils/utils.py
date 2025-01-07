import numpy as np
import cv2
import os

def fast_clip(mat, low, high):
    mat[np.greater(mat,high)] = high
    mat[np.less(mat,low)] = low
    return mat

def get_roi_from_region(region: dict):
    xs = []
    ys = []

    if region.get("shape_attributes").get("name") == "polygon":
        xs = region.get("shape_attributes").get("all_points_x")
        ys = region.get("shape_attributes").get("all_points_y")

    if region.get("shape_attributes").get("name") == "rect":
        x = region.get("shape_attributes").get("x")
        y = region.get("shape_attributes").get("y")
        w = region.get("shape_attributes").get("width")
        h = region.get("shape_attributes").get("height")
        xs = [x, x + w, x + w, x]
        ys = [y, y, y + h, y + h]

    return np.array(list(zip(xs, ys)))

def polygon_area(points):
    def det(P0, P1):
        return P0[0]*P1[1]-P0[1]*P1[0]
    
    result = 0
    for i in range(len(points)-1):
        result+=det(points[i], points[i+1])
    result+=det(points[-1], points[0])
    return abs(result)/2

def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def get_percentile_mask(img, polygon, p_value): 
    if polygon is not None:   
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [np.int32(polygon)], 1)
    else:
        mask = None
    hist = cv2.calcHist([img], [0], mask, [256], [0,256])
    num_pixels = np.sum(hist)
    value = np.argmax(cv2.compare(cv2.integral(hist)[1:,1]/num_pixels, p_value/100, cmpop=cv2.CMP_GE))
    return value

def warp_polygon_points(polygon, M):
    num_points = len(polygon)
    polygon_c = np.ones((num_points,3))
    polygon_c[:, :2] = polygon
    return np.matmul(polygon_c, M.T)

def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def max_distance_from_center(polygon):
    max_distance=0
    center = np.sum(polygon, axis=0)/len(polygon)
    for point in polygon:
        segment = point_distance(center, point)
        if segment > max_distance:
            max_distance = segment
    return max_distance

def safe_make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)