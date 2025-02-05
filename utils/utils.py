import cv2
import os
import re
import subprocess
import numpy as np
import numba as nb
from time import perf_counter_ns

import sys
sys.path.insert(0, './im2col_2D/python/')
from im2col import im2col_SIMD as im2col

interpolations = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
    'area': cv2.INTER_AREA
}

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

def im2col_SIMD(img, ker_h, ker_w):
    return im2col(img, ker_h, ker_w)

@nb.jit(cache=True, nopython=True)
def indexing(arr, bool_mask):
    return arr[bool_mask]

@nb.jit(cache=True, nopython=True)
def indexing_assignment(arr_left, arr_right, bool_mask):
    arr_left[bool_mask] = arr_right

@nb.jit(cache=True, nopython=True, fastmath=True)
def gt_wth_counter(arr, th):
    indexes = np.greater(arr[:,-1], th)
    return indexes, indexes.sum()

@nb.jit(nopython=True, cache=True)  # Enable nopython mode for faster performance
def col2img(outImg, colTensor):
    H, W = outImg.shape
    H = H // 2
    W = W // 2

    # Fill in the output image
    for i in range(H):
        for j in range(W):
            outImg[2 * i, 2 * j] = colTensor[i*W+j, 0]
            outImg[2 * i, 2 * j + 1] = colTensor[i*W+j, 1]
            outImg[2 * i + 1, 2 * j] = colTensor[i*W+j, 2]
            outImg[2 * i + 1, 2 * j + 1] = colTensor[i*W+j, 3]
    return outImg

# Optimized Version
def interativeProcessing(H,W, inputVec, interpreters, th, nch=49):
    total_NN_time = 0
    total_data_time = 0
    n = len(inputVec)
    list_of_indexes = []
    outputs = []

    ###########################################################################
    ###########   Run First Interation on all inputs ##########################
    ###########################################################################

    interpreters[0].resize_tensor_input(0, [n,nch], strict=True)
    interpreters[0].allocate_tensors()
    input_details = interpreters[0].get_input_details()
    output_details = interpreters[0].get_output_details()
    
    start = perf_counter_ns()
    interpreters[0].set_tensor(input_details[0]['index'], inputVec)
    interpreters[0].invoke()
    embedding = interpreters[0].get_tensor(output_details[0]['index'])
    outputs.append(interpreters[0].get_tensor(output_details[1]['index']))
    total_NN_time+= (perf_counter_ns()-start)

    ###########################################################################
    ######  Perform additional iterations only on the patches    ##############
    ######  where the predicted variance is over the threshold   ##############
    ###########################################################################

    for i in range(1,len(interpreters)):
        start = perf_counter_ns()
        indexes, num_indexes = gt_wth_counter(outputs[-1], th[i-1])
        list_of_indexes.append(indexes)
        embed_ch = embedding.shape[-1]
        total_data_time+= (perf_counter_ns()-start)
        interpreters[i].resize_tensor_input(0, [num_indexes,nch], strict=True)
        interpreters[i].resize_tensor_input(1, [num_indexes,embed_ch], strict=True)
        interpreters[i].allocate_tensors()
        input_details = interpreters[i].get_input_details()
        output_details = interpreters[i].get_output_details()
        
        start = perf_counter_ns()
        i_vec = indexing(inputVec, list_of_indexes[-1])
        e_vec = indexing(embedding, list_of_indexes[-1])
        total_data_time+= (perf_counter_ns()-start)
        start = perf_counter_ns()
        interpreters[i].set_tensor(input_details[0]['index'], i_vec)
        interpreters[i].set_tensor(input_details[1]['index'], e_vec)
        interpreters[i].invoke()
        if i == len(interpreters)-1:
            outputs.append(interpreters[i].get_tensor(output_details[0]['index']))
        else:
            embedding = interpreters[i].get_tensor(output_details[0]['index'])
            outputs.append(interpreters[i].get_tensor(output_details[1]['index']))
        total_NN_time += perf_counter_ns()-start
    
    ###########################################################################
    ###########   Build the final image from the output patches ###############
    ###########################################################################

    start = perf_counter_ns()
    for i in range(len(outputs)-1,0,-1):
        indexing_assignment(outputs[i-1][:,0:4], outputs[i][:,0:4], list_of_indexes[i-1])

    result = np.empty((H*2,W*2), dtype=np.float32)
    result = col2img(result, outputs[0])
    total_data_time+= (perf_counter_ns()-start)
    return result, total_NN_time/1e6, total_data_time/1e6

def iterativeUpscale(img, interpreters, th, ker=(7,7)):
    H,W = img.shape
    nch = ker[0]*ker[1]
    start = perf_counter_ns()
    reshaped_input = im2col_SIMD(img, ker[0], ker[1])
    im2col_time = (perf_counter_ns()-start)/1e6
    up_image, total_nn_time, total_data_time = interativeProcessing(H,W, reshaped_input, interpreters, th, nch=nch)
    return up_image, total_nn_time, total_data_time+im2col_time