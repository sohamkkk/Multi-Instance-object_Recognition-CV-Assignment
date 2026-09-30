"""
Multi-Instance Object Recognition in Cluttered Scenes
Pipeline: SIFT (OpenCV) -> vectorized from-scratch ratio-test matching -> from-scratch
4D Generalized Hough Transform -> from-scratch RANSAC + least-squares affine
fit -> greedy multi-instance extraction -> polygon-IoU NMS.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

RATIO_THRESHOLD = 0.75
HOUGH_MIN_VOTES = 3
HOUGH_BIN_XY = 50
HOUGH_BIN_SCALE = 0.1
HOUGH_BIN_ANGLE = 30
RANSAC_ITERS = 1000
RANSAC_INLIER_PX = 5.0
MIN_INLIERS = HOUGH_MIN_VOTES  
MIN_INLIER_RATIO = 0.4
MAX_MEAN_REPROJ_ERROR = 3.0
NMS_IOU_THRESHOLD = 0.3

def extract_features(image_path, max_width=2400, max_features=8000):
    """Extracts SIFT features using OpenCV with limits to prevent hanging.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
        
    h, w = img.shape
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
        
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    if descriptors is None:
        descriptors = np.array([])
        keypoints = ()
        
    return img, keypoints, descriptors

def match_features_from_scratch(desc_scene, desc_template, ratio_threshold=RATIO_THRESHOLD,
                                 chunk_size=500):
    """Computes Euclidean distance and applies Lowe's ratio test from scratch (Vectorized).
    """
    matches = []
    if len(desc_scene) == 0 or len(desc_template) == 0:
        return matches

    template_sq = np.sum(desc_template**2, axis=1)

    for start in range(0, len(desc_scene), chunk_size):
        chunk = desc_scene[start:start + chunk_size]
        sq_dists = (np.sum(chunk**2, axis=1)[:, np.newaxis] +
                    template_sq -
                    2 * np.dot(chunk, desc_template.T))
        distances = np.sqrt(np.maximum(sq_dists, 0))

        for j in range(len(chunk)):
            sorted_indices = np.argsort(distances[j])
            best_idx = sorted_indices[0]
            second_best_idx = sorted_indices[1]

            if distances[j, second_best_idx] == 0:
                continue
            if distances[j, best_idx] / distances[j, second_best_idx] < ratio_threshold:
                matches.append((start + j, best_idx))

    return matches

def generalized_hough_transform(matches, kp_scene, kp_template, template_shape):
    hough_space = {}
    h, w = template_shape
    cx, cy = w / 2, h / 2

    for scene_idx, temp_idx in matches:
        ks = kp_scene[scene_idx]
        kt = kp_template[temp_idx]

        scale = ks.size / kt.size
        angle_deg = (ks.angle - kt.angle) % 360.0
        theta = math.radians(angle_deg)

        dx = cx - kt.pt[0]
        dy = cy - kt.pt[1]

        pred_cx = ks.pt[0] + scale * (dx * math.cos(theta) - dy * math.sin(theta))
        pred_cy = ks.pt[1] + scale * (dx * math.sin(theta) + dy * math.cos(theta))

        bin_x = int(pred_cx // HOUGH_BIN_XY) * HOUGH_BIN_XY
        bin_y = int(pred_cy // HOUGH_BIN_XY) * HOUGH_BIN_XY
        bin_s = round(scale / HOUGH_BIN_SCALE) * HOUGH_BIN_SCALE
        bin_theta = int(angle_deg // HOUGH_BIN_ANGLE) * HOUGH_BIN_ANGLE

        bin_key = (bin_x, bin_y, bin_s, bin_theta)
        if bin_key not in hough_space:
            hough_space[bin_key] = []
        hough_space[bin_key].append((scene_idx, temp_idx))

    return {k: v for k, v in hough_space.items() if len(v) >= HOUGH_MIN_VOTES}

def solve_affine_least_squares(pts_scene, pts_template):
    A = []
    b = []
    for (xs, ys), (xt, yt) in zip(pts_scene, pts_template):
        A.append([xt, yt, 1, 0, 0, 0])
        A.append([0, 0, 0, xt, yt, 1])
        b.append([xs])
        b.append([ys])

    A = np.array(A)
    b = np.array(b)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x.reshape(2, 3)

def is_collinear(pts, eps=15.0):
    (x1, y1), (x2, y2), (x3, y3) = pts
    twice_area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return twice_area < eps


def is_degenerate_affine(affine_mat, max_axis_ratio=3.0, max_shear_deg=35.0, min_det=1e-8):
    """Sanity-check the fitted affine's linear part WITHOUT assuming any
    particular absolute scale
    """
    a, b = affine_mat[0, 0], affine_mat[0, 1]
    c, d = affine_mat[1, 0], affine_mat[1, 1]
    col1 = np.array([a, c])  # where the template's x-axis lands
    col2 = np.array([b, d])  # where the template's y-axis lands
    n1, n2 = np.linalg.norm(col1), np.linalg.norm(col2)
    det = a * d - b * c

    if n1 < 1e-9 or n2 < 1e-9 or abs(det) < min_det:
        return True  # singular / collapsed to a line or point

    if max(n1, n2) / min(n1, n2) > max_axis_ratio:
        return True  # axes scaled very differently -> stretch/shear

    cos_angle = abs(np.dot(col1, col2) / (n1 * n2))
    if cos_angle > math.sin(math.radians(max_shear_deg)):
        return True  # axes far from perpendicular -> shear

    return False

def compute_mean_reprojection_error(affine_mat, inliers, kp_scene, kp_template):
    errors = []
    for scene_idx, temp_idx in inliers:
        pt_t = np.array([kp_template[temp_idx].pt[0], kp_template[temp_idx].pt[1], 1.0])
        pred_s = affine_mat @ pt_t
        pt_s = np.array(kp_scene[scene_idx].pt)
        errors.append(np.linalg.norm(pt_s - pred_s))
    return float(np.mean(errors)) if errors else float("inf")

def ransac_affine(matches, kp_scene, kp_template, iters=RANSAC_ITERS, threshold=RANSAC_INLIER_PX):
    best_inliers = []
    best_affine = None

    if len(matches) < 3:
        return None, []

    pts_s = np.float32([kp_scene[m[0]].pt for m in matches])
    pts_t = np.float32([kp_template[m[1]].pt for m in matches])

    for _ in range(iters):
        indices = np.random.choice(len(matches), 3, replace=False)
        sample_s = pts_s[indices]
        sample_t = pts_t[indices]

        if is_collinear(sample_t):
            continue

        affine_mat = solve_affine_least_squares(sample_s, sample_t)

        if is_degenerate_affine(affine_mat):
            continue

        inliers = []
        for i in range(len(matches)):
            pt_t = np.array([pts_t[i][0], pts_t[i][1], 1.0])
            pred_s = affine_mat @ pt_t
            error = np.linalg.norm(pts_s[i] - pred_s)

            if error < threshold:
                inliers.append(matches[i])

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_affine = solve_affine_least_squares(
                np.float32([kp_scene[m[0]].pt for m in inliers]),
                np.float32([kp_template[m[1]].pt for m in inliers])
            )

    return best_affine, best_inliers

def polygon_area(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def ensure_ccw(pts):
    signed_area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        signed_area += (x1 * y2 - x2 * y1)
    if signed_area < 0:
        return pts[::-1]
    return pts

def _line_intersect(p1, p2, a, b):
    A1, B1 = b[1] - a[1], a[0] - b[0]
    C1 = A1 * a[0] + B1 * a[1]
    A2, B2 = p2[1] - p1[1], p1[0] - p2[0]
    C2 = A2 * p1[0] + B2 * p1[1]
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-9:
        return p2
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return (x, y)

def clip_polygon(subject, clip):
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    output = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        input_list, output = output, []
        if not input_list:
            break
        for j in range(len(input_list)):
            cur, prev = input_list[j], input_list[j - 1]
            cur_in, prev_in = inside(cur, a, b), inside(prev, a, b)
            if cur_in:
                if not prev_in:
                    output.append(_line_intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(_line_intersect(prev, cur, a, b))
    return output

def compute_iou(boxA, boxB):
    polyA = ensure_ccw([tuple(p) for p in boxA[:4]])
    polyB = ensure_ccw([tuple(p) for p in boxB[:4]])

    inter_pts = clip_polygon(polyA, polyB)
    inter_area = polygon_area(np.array(inter_pts)) if len(inter_pts) >= 3 else 0.0

    areaA = polygon_area(np.array(polyA))
    areaB = polygon_area(np.array(polyB))
    union = areaA + areaB - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union

def draw_and_suppress(scene_img, template_img, detected_instances):
    h, w = template_img.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h], [0, 0]])
    final_boxes = []

    for affine_mat in detected_instances:
        projected_corners = np.zeros((5, 2))
        for i in range(5):
            pt = np.array([corners[i][0], corners[i][1], 1.0])
            projected_corners[i] = affine_mat @ pt

        is_duplicate = False
        for existing_box in final_boxes:
            if compute_iou(projected_corners, existing_box) > NMS_IOU_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            final_boxes.append(projected_corners)

    result_img = cv2.cvtColor(scene_img, cv2.COLOR_GRAY2BGR)
    for box in final_boxes:
        for i in range(4):
            pt1 = (int(box[i][0]), int(box[i][1]))
            pt2 = (int(box[i + 1][0]), int(box[i + 1][1]))
            cv2.line(result_img, pt1, pt2, (0, 255, 0), 3)

    return result_img, final_boxes

def draw_naive_matches(scene_img, template_img, kp_scene, kp_template, matches):
    h_t, w_t = template_img.shape
    h_s, w_s = scene_img.shape
    h = max(h_t, h_s)

    canvas = np.zeros((h, w_t + w_s, 3), dtype=np.uint8)
    canvas[:h_t, :w_t] = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
    canvas[:h_s, w_t:w_t + w_s] = cv2.cvtColor(scene_img, cv2.COLOR_GRAY2BGR)

    rng = np.random.default_rng(42)
    for scene_idx, temp_idx in matches:
        pt_t = np.round(kp_template[temp_idx].pt).astype(int)
        pt_s = np.round(kp_scene[scene_idx].pt).astype(int) + np.array([w_t, 0])
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        cv2.line(canvas, tuple(pt_t), tuple(pt_s), color, 1)
        cv2.circle(canvas, tuple(pt_t), 3, color, -1)
        cv2.circle(canvas, tuple(pt_s), 3, color, -1)

    return canvas

def main():
    # Input location of the images
    template_img, kp_t, desc_t = extract_features("all-drunk-remove.png")
    scene_img, kp_s, desc_s = extract_features("Cluttered_Table.jpeg")

    raw_matches = match_features_from_scratch(desc_s, desc_t)
    print(f"Raw ratio-test matches: {len(raw_matches)}")

    naive_img = draw_naive_matches(scene_img, template_img, kp_s, kp_t, raw_matches)
    cv2.imwrite("naive_matches.png", naive_img)

    hough_clusters = generalized_hough_transform(raw_matches, kp_s, kp_t, template_img.shape)
    print(f"Hough bins with >= {HOUGH_MIN_VOTES} votes: {len(hough_clusters)}")

    detected_instances = []
    remaining_matches = set(raw_matches)

    for bin_key, cluster_matches in sorted(hough_clusters.items(), key=lambda x: len(x[1]), reverse=True):
        valid_cluster = [m for m in cluster_matches if m in remaining_matches]
        if len(valid_cluster) < HOUGH_MIN_VOTES:
            continue

        affine_mat, inliers = ransac_affine(valid_cluster, kp_s, kp_t)
        if affine_mat is None:
            continue

        inlier_ratio = len(inliers) / len(valid_cluster)
        mean_err = compute_mean_reprojection_error(affine_mat, inliers, kp_s, kp_t)

        if (len(inliers) >= MIN_INLIERS
                and inlier_ratio >= MIN_INLIER_RATIO
                and mean_err <= MAX_MEAN_REPROJ_ERROR):
            detected_instances.append(affine_mat)
            remaining_matches -= set(inliers)

    print(f"Verified instances before duplicate-suppression: {len(detected_instances)}")
    # Several Hough bins can independently re-discover the same physical instance

    final_image, final_boxes = draw_and_suppress(scene_img, template_img, detected_instances)
    print(f"Final instances after NMS: {len(final_boxes)}")
    cv2.imwrite("final_detections.png", final_image)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(cv2.cvtColor(naive_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Naive matching ({len(raw_matches)} raw matches)")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Final detections ({len(final_boxes)} instances)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()