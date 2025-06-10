import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

import ipywidgets as widgets

from itertools import combinations, product

import random

from collections import Counter

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

from sklearn.metrics.pairwise import cosine_similarity
from tkinter import Tk, Button, Label, filedialog, Frame, Canvas, Scrollbar, VERTICAL, RIGHT, LEFT, Y, BOTH

from PIL import Image, ImageTk
import os

def get_contours_by_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_cnt = None
    C = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 30:
            C.append(cnt)
    print(len(C))
    return C

def auto_white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    
    a = a - ((avg_a - 128) * (l / 255.0) * 1.1)
    b = b - ((avg_b - 128) * (l / 255.0) * 1.1)
    
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    
    balanced_lab = cv2.merge([l, a, b])
    return cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2BGR)

def get_all_rois(img, contours, max_cols=4, figsize=(12, 8)):
    n = len(contours)
    MIN_AREA = 100 * 150
    rois = []
    if n == 0:
        print("Нет контуров для отображения")
        return

    n_rows = (n + max_cols - 1) // max_cols
    n_cols = min(n, max_cols)
    
    plt.figure(figsize=figsize)
    
    for i, cont in enumerate(contours, 1):
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(m, [cont], -1, 255, thickness=-1)
        masked_img = cv2.bitwise_and(img, img, mask=m)
        
        x, y, w, h = cv2.boundingRect(cont)
        roi = masked_img[y:y+h, x:x+w].copy()
        
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        rh, rw = roi.shape[:2]
        if rh < 50 or h < 50:
            print("SMALL 1")
            continue
        elif rh * rw < MIN_AREA * 0.7:
            print("SMALL (AREA) 2")
            continue
        else:
            rois.append(roi)
        
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(roi)
        plt.axis('off')
        plt.title(f'ROI {i}')

    plt.tight_layout()
    plt.show()
    return rois

def get_all_rois(img, contours, max_cols=4, figsize=(12, 8)):
    n = len(contours)
    MIN_AREA = 100 * 150
    rois = []
    if n == 0:
        print("Нет контуров для отображения")
        return

    n_rows = (n + max_cols - 1) // max_cols
    n_cols = min(n, max_cols)
    
    plt.figure(figsize=figsize)
    
    for i, cont in enumerate(contours, 1):
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(m, [cont], -1, 255, thickness=-1)
        masked_img = cv2.bitwise_and(img, img, mask=m)
        
        x, y, w, h = cv2.boundingRect(cont)
        roi = masked_img[y:y+h, x:x+w].copy()
        
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        rh, rw = roi.shape[:2]
        if rh < 50 or h < 50:
            print("SMALL 1")
            continue
        elif rh * rw < MIN_AREA * 0.7:
            print("SMALL (AREA) 2")
            continue
        else:
            rois.append(roi)
        
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(roi)
        plt.axis('off')
        plt.title(f'ROI {i}')

    plt.tight_layout()
    plt.show()
    return rois

def get_all_rois_lite(img, contours, max_cols=4, figsize=(12, 8)):
    min_area = 150*100*0.3
    n = len(contours)
    if n == 0:
        print("Нет контуров для отображения")
        return
    rois = []
    
    # plt.figure(figsize=figsize)
    
    n_rows = (n + max_cols - 1) // max_cols
    n_cols = min(n, max_cols)
    
    for i, cnt in enumerate(contours, 1):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
        masked = cv2.bitwise_and(img, img, mask=mask)

        x, y, w, h = cv2.boundingRect(cnt)
        roi = masked[y:y+h, x:x+w]

        # if h < 50 or w < 50:
        #     continue
        # if h * w < min_area * 0.7:
        #     continue

        rois.append(roi)
        
        # plt.subplot(n_rows, n_cols, i)
        # plt.imshow(roi)
        # plt.axis('off')
        # plt.title(f'ROI {i}')

    # plt.tight_layout()
    # plt.show()
    return rois

def rebalance_image(img, method='stretch'):
    if method == 'stretch':
        non_zero = img[img > 0]
        if non_zero.size == 0:
            return img
        min_val = np.min(non_zero)
        max_val = np.max(img)
        if max_val > min_val:
            stretched = (img.astype(np.float32) - min_val) / (max_val - min_val) * 255
            stretched = np.clip(stretched, 0, 255).astype(np.uint8)
            stretched = np.where(img == 0, 0, stretched)
            return stretched
        return img

    elif method == 'histogram_eq':
        mask = img > 0
        if np.sum(mask) == 0:
            return img
        temp = img.copy()
        min_non_zero = np.min(temp[mask])
        temp[~mask] = min_non_zero
        eq = cv2.equalizeHist(temp.astype(np.uint8))
        eq[~mask] = 0
        return eq

    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img.astype(np.uint8))

    elif method == 'power':
        # Усиление яркого — подавление слабого (power-law)
        img_float = img.astype(np.float32) / 255.0
        boosted = np.power(img_float, 2.2)  # γ = 2.2, можно настраивать
        boosted = np.clip(boosted * 255, 0, 255).astype(np.uint8)
        return boosted

    return img

def adaptive_hist_knee_threshold_multi(
    img,
    passes=2,
    min_threshold=0.005,
    aggressive_pass=True,
    rebalance_method='stretch',  # 'stretch', 'histogram_eq', 'clahe'
    plot=False,
    return_knees=False
):
    """
    Многоцикловый адаптивный knee-based порог с перебалансировкой яркости.
    """
    def find_knee(channel, threshold):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
        cdf = hist.cumsum()
        if cdf[-1] == 0:
            return None, hist, None, None, None
        cdf_norm = cdf / cdf[-1]
        grad = np.gradient(cdf_norm)
        second_grad = np.gradient(grad)
        for i in range(5, len(second_grad) - 5):
            if second_grad[i] > threshold and grad[i] > threshold:
                return i, hist, cdf_norm, grad, second_grad
        return int(np.argmax(grad[5:-5])) + 5, hist, cdf_norm, grad, second_grad

    original = img.copy()
    current = img.copy()
    knees = []
    images = [original]
    hists, cdfs, grads, s_grads = [], [], [], []

    for i in range(passes):
        t = min_threshold * (2 if aggressive_pass else 1.5) ** i
        knee, hist, cdf, grad, s_grad = find_knee(current, t)
        if knee is None:
            print(f"[{i+1}] Пропуск: не найден knee.")
            break
        knees.append(knee)
        hists.append(hist)
        cdfs.append(cdf)
        grads.append(grad)
        s_grads.append(s_grad)
        mask = current >= knee
        current = np.where(mask, current, 0).astype(np.uint8)
        current = rebalance_image(current, rebalance_method)
        current = cv2.GaussianBlur(current, (5, 5), 0)
        # current = np.where(current >= knee, current, 0).astype(np.uint8)
        images.append(current)

    # итоговая бинаризация
    final = np.where(current > 0, 255, 0).astype(np.uint8)

    if plot:
        fig, axs = plt.subplots(nrows=3, ncols=passes, figsize=(5 * passes, 10))
        for i in range(passes):
            # изображения
            axs[0, i].imshow(255-images[i + 1], cmap='gray')
            axs[0, i].set_title(f'Pass {i+1}\nknee={knees[i]}')
            axs[0, i].axis('off')
            # гистограммы
            axs[1, i].plot(hists[i])
            axs[1, i].axvline(knees[i], color='red', linestyle='--')
            axs[1, i].set_title(f'Histogram {i+1}')
            axs[1, i].set_yscale('log')
            # производные
            axs[2, i].plot(grads[i], label='1st deriv')
            axs[2, i].plot(s_grads[i], label='2nd deriv')
            axs[2, i].axvline(knees[i], color='red', linestyle='--')
            axs[2, i].legend()
            axs[2, i].set_title(f'Derivatives {i+1}')
        plt.tight_layout()
        plt.show()

    if return_knees:
        return final, knees
    return final

def show_image_on_label(label, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((400, 300))
    img_tk = ImageTk.PhotoImage(img_pil)
    label.imgtk = img_tk
    label.config(image=img_tk)

def show_rois_in_gallery(rois, container):
    for widget in container.winfo_children():
        widget.destroy()

    canvas = Canvas(container)
    scrollbar = Scrollbar(container, orient=VERTICAL, command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.pack(side=RIGHT, fill=Y)

    max_cols = 4
    for idx, roi in enumerate(rois):
        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((100, 70))
        img_tk = ImageTk.PhotoImage(img_pil)

        row, col = divmod(idx, max_cols)
        lbl = Label(scrollable_frame, image=img_tk)
        lbl.image = img_tk
        lbl.grid(row=row, column=col, padx=5, pady=5)


def process_roi_initial(roi):
    roi = auto_white_balance(roi)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # top-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(gray, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)
    ########

    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
    mean, stddev = cv2.meanStdDev(enhanced)
    brightness = mean[0][0]
    contrast = stddev[0][0]

    if brightness < 100 and contrast < 50:
        print("UP & UP")
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=0)

    closed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, closed_kernel, iterations=2)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

    masked = adaptive_hist_knee_threshold_multi(enhanced, 
                                                passes=2,
                                                aggressive_pass=True,
                                                plot=False,
                                                rebalance_method='power')

    contik, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.array([cv2.contourArea(cnt) for cnt in contik])
    if len(areas) >= 2:
        log_areas = np.log1p(areas).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, n_init='auto', random_state=0).fit(log_areas)
        labels = kmeans.labels_
        big_cluster = np.argmax([areas[labels == i].mean() for i in range(2)])
        contours_big = [cnt for i, cnt in enumerate(contik) if labels[i] == big_cluster]
    else:
        contours_big = contik
        print("DEBUG not enoght areas for clustering")

    contik_filtered = [cnt for cnt in contours_big if cv2.contourArea(cnt) > 800]
    i_rois = get_all_rois_lite(roi, contik_filtered, max_cols=1)
    return i_rois


def process_individual_card(i_roi):
    i_roi = auto_white_balance(i_roi)
    gray = cv2.cvtColor(i_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
    gray = clahe.apply(gray)

    # top-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(gray, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)
    enhanced = rebalance_image(enhanced, 'power')
    ########

    closed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, closed_kernel, iterations=1)

    masked = adaptive_hist_knee_threshold_multi(
        enhanced, passes=2, aggressive_pass=True, plot=False, rebalance_method='power')

    contik, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contik_filtered = [cnt for cnt in contik if cv2.contourArea(cnt) > 1200]
    return len(contik_filtered)

def main_pipeline(img):
    img = auto_white_balance(img)
    lower_blue = np.array([80, 40, 40])
    upper_blue = np.array([130, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    base_mask = mask

    C = get_contours_by_mask(base_mask)
    ii = img.copy()
    rois = get_all_rois(ii, C)
    all_i_rois = []
    total_count = 0

    for roi in rois:
        i_rois = process_roi_initial(roi)
        all_i_rois.extend(i_rois)
        print("DEBUG len(i_rois): ", len(i_rois))
        for i_roi in i_rois:
            total_count += process_individual_card(i_roi)
    return total_count, all_i_rois


def load_and_process():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    count, i_rois = main_pipeline(img)
    label_result.config(text=f"Number of cards: {count}")
    show_image_on_label(label_image, img)
    show_rois_in_gallery(i_rois, gallery_frame)


# GUI setup
root = Tk()
root.title("Card Counter")

btn_load = Button(root, text="Choose the file", command=load_and_process)
btn_load.pack(pady=10)

label_result = Label(root, text="")
label_result.pack(pady=5)

label_image = Label(root)
label_image.pack(pady=5)

gallery_frame = Frame(root, height=200)
gallery_frame.pack(fill=BOTH, expand=True, pady=5)

root.mainloop()
