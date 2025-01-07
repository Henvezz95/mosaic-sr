import cv2
import os
import re
import subprocess
import numpy as np

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


def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu