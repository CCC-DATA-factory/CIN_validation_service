import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import psutil
import uvicorn
import time

import os


# --- Our CINValidator Class ---
class CINValidator:
    def __init__(self, template_path, min_good_matches=120, inlier_threshold=120, ratio_threshold=0.7):
        """
        Initialize the validator with a template image of the CIN.
        :param template_path: Path to the CIN template image.
        :param min_good_matches: Minimum number of good matches required.
        :param inlier_threshold: Minimum number of inlier matches (from homography) required.
        :param ratio_threshold: Lowe's ratio threshold for filtering matches.
        """
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError(f"Template image could not be loaded from path: {template_path}")
        
        self.sift = cv2.SIFT_create()
        
        self.kp_template, self.des_template = self.sift.detectAndCompute(self.template, None)
        if self.des_template is None:
            raise ValueError("No descriptors found in the template image.")
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.min_good_matches = min_good_matches
        self.inlier_threshold = inlier_threshold
        self.ratio_threshold = ratio_threshold

    def validate_cin(self, input_img):
        """
        Validate whether the input image (grayscale) contains the CIN card.
        Returns a tuple: (detected: bool, score: dict)
        The score dict includes: number of good matches, inlier count, and inlier ratio.
        Additionally returns time consumed, CPU, and memory usage in a readable format.
        :param input_img: The input image as a grayscale numpy array.
        :return: (True, score, time_elapsed, cpu_usage, memory_usage) if CIN card detected, (False, score, time_elapsed, cpu_usage, memory_usage) otherwise.
        """
        start_time = time.time()
        process = psutil.Process()
        start_cpu = process.cpu_percent()
        start_memory = process.memory_info().rss

        kp_input, des_input = self.sift.detectAndCompute(input_img, None)
        if des_input is None:
            return False, {"good_matches": 0, "inliers": 0, "inlier_ratio": 0.0}, "0 seconds", "0 cores", "0 MB"

        try:
            matches = self.flann.knnMatch(self.des_template, des_input, k=2)
        except Exception as e:
            print("Error during feature matching:", e)
            return False, {"good_matches": 0, "inliers": 0, "inlier_ratio": 0.0}, "0 seconds", "0 cores", "0 MB"

        good_matches = [m for m, n in matches if m.distance < self.ratio_threshold * n.distance]
        num_good = len(good_matches)

        if num_good < self.min_good_matches:
            return False, {"good_matches": num_good, "inliers": 0, "inlier_ratio": 0.0}, "0 seconds", "0 cores", "0 MB"

        src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_input[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None or mask is None:
            return False, {"good_matches": num_good, "inliers": 0, "inlier_ratio": 0.0}, "0 seconds", "0 cores", "0 MB"

        inliers = int(np.sum(mask))
        ratio = float(inliers) / num_good if num_good > 0 else 0.0
        detected = inliers > self.inlier_threshold

        score = {"good_matches": num_good, "inliers": inliers, "inlier_ratio": ratio}
        
        end_time = time.time()
        time_elapsed = end_time - start_time
        end_cpu = process.cpu_percent()
        end_memory = process.memory_info().rss
        cpu_usage = end_cpu - start_cpu
        memory_usage = abs((end_memory - start_memory) / (1024 * 1024))  # Convert bytes to MB

        return detected, score, f"{time_elapsed:.2f} seconds", f"{cpu_usage:.2f}% of 1 core", f"{memory_usage:.2f} MB"

    def get_bbox(self, input_img):
        """
        If the CIN card is detected, compute and return the bounding box of the CIN.
        :param input_img: The input image as a grayscale numpy array.
        :return: Bounding box as a numpy array with shape (4, 1, 2) if detected, otherwise None.
        """
        kp_input, des_input = self.sift.detectAndCompute(input_img, None)
        if des_input is None:
            return None
        
        try:
            matches = self.flann.knnMatch(self.des_template, des_input, k=2)
        except Exception as e:
            print("Error during feature matching:", e)
            return None
        
        good_matches = [m for m, n in matches if m.distance < self.ratio_threshold * n.distance]
        if len(good_matches) < self.min_good_matches:
            return None
        
        src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_input[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None or mask is None:
            return None
        inliers = int(np.sum(mask))
        if inliers < self.inlier_threshold:
            return None
        
        h, w = self.template.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        bbox = cv2.perspectiveTransform(pts, M)
        return bbox

# --- FastAPI App ---
app = FastAPI()

TEMPLATE_FRONT_PATH = os.getenv("TEMPLATE_FRONT_PATH","templates/front.jpeg")
TEMPLATE_BACK_PATH = os.getenv("TEMPLATE_BACK_PATH",'templates/back.jpeg')
MIN_GOOD_MATCHES = int(os.getenv("MIN_GOOD_MATCHES", 120))
INLIER_THRESHOLD_FRONT = int(os.getenv("INLIER_THRESHOLD_FRONT", 120))
INLIER_THRESHOLD_BACK = int(os.getenv("INLIER_THRESHOLD_BACK", 50))
RATIO_THRESHOLD = float(os.getenv("RATIO_THRESHOLD", 0.7))

try:
    front_validator = CINValidator(TEMPLATE_FRONT_PATH, min_good_matches=MIN_GOOD_MATCHES, inlier_threshold=INLIER_THRESHOLD_FRONT, ratio_threshold=RATIO_THRESHOLD)
    back_validator = CINValidator(TEMPLATE_BACK_PATH, min_good_matches=MIN_GOOD_MATCHES, inlier_threshold=INLIER_THRESHOLD_BACK, ratio_threshold=RATIO_THRESHOLD)
except Exception as e:
    print("Error initializing validators:", e)
    raise

def load_image_from_bytes(file_bytes: bytes, grayscale: bool = True):
    nparr = np.frombuffer(file_bytes, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(nparr, flag)
    return img

def crop_bbox(image: np.ndarray, bbox: np.ndarray):
    pts = bbox.reshape(4, 2)
    x_min, y_min = np.int32(pts.min(axis=0))
    x_max, y_max = np.int32(pts.max(axis=0))
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, image.shape[1])
    y_max = min(y_max, image.shape[0])
    return image[y_min:y_max, x_min:x_max]

@app.post("/validate/front")
async def validate_front(file: UploadFile = File(...)):
    contents = await file.read()
    img = load_image_from_bytes(contents, grayscale=True)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file for front CIN.")
    detected, score, time_elapsed, cpu_usage, memory_usage = front_validator.validate_cin(img)
    return JSONResponse(content={"front_detected": bool(detected), "score": score,'metrics':{"inferance_time":time_elapsed,'cpu_usage':cpu_usage,'memory_usage':memory_usage}})

@app.post("/validate/back")
async def validate_back(file: UploadFile = File(...)):
    contents = await file.read()
    img = load_image_from_bytes(contents, grayscale=True)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file for back CIN.")
    detected, score, time_elapsed, cpu_usage, memory_usage = back_validator.validate_cin(img)
    return JSONResponse(content={"back_detected": bool(detected), "score": score , 'metrics':{"inferance_time":time_elapsed,'cpu_usage':cpu_usage,'memory_usage':memory_usage}})

@app.post("/croped/front")
async def get_front_cropped(file: UploadFile = File(...)):

    contents = await file.read()
    img_color = load_image_from_bytes(contents, grayscale=False)
    if img_color is None:
        raise HTTPException(status_code=400, detail="Invalid image file for front CIN.")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    bbox = front_validator.get_bbox(img_gray)
    if bbox is None:
        raise HTTPException(status_code=404, detail="Front CIN card not detected.")
    crop = crop_bbox(img_color, bbox)
    is_success, buffer = cv2.imencode(".jpg", crop)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode cropped image.")
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.post("/croped/back")
async def get_back_cropped(file: UploadFile = File(...)):
    contents = await file.read()
    img_color = load_image_from_bytes(contents, grayscale=False)
    if img_color is None:
        raise HTTPException(status_code=400, detail="Invalid image file for back CIN.")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    bbox = back_validator.get_bbox(img_gray)
    if bbox is None:
        raise HTTPException(status_code=404, detail="Back CIN card not detected.")
    crop = crop_bbox(img_color, bbox)
    is_success, buffer = cv2.imencode(".jpg", crop)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode cropped image.")
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8082)
