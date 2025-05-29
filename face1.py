import cv2
import numpy as np
import mediapipe as mp
import sys

def extract_landmarks(image, face_mesh, max_num_faces=1):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    h, w = image.shape[:2]
    points = []
    for lm in landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
    return points

def rect_to_bb(rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    return (x, y, w, h)

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2_rect_int.append((int(t2[i][0] - r2[0]), int(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply affine transformation to patch
    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask

    # Copy patch to destination image
    img2_part = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    img2_part = img2_part * ((1.0, 1.0, 1.0) - mask)
    img2_part = img2_part + img2_rect
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_part

def get_face_convex_hull(points):
    hull_index = cv2.convexHull(np.array(points), returnPoints=False)
    hull = []
    for idx in hull_index:
        hull.append(points[idx[0]])
    return hull, hull_index

def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True

def calculate_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()
    delaunayTri = []

    pt = []
    for t in triangleList:
        pt.append((int(t[0]), int(t[1])))
        pt.append((int(t[2]), int(t[3])))
        pt.append((int(t[4]), int(t[5])))

        pts = [pt[0], pt[1], pt[2]]
        pt = []

        if all(rect_contains(rect, p) for p in pts):
            ind = []
            for p in pts:
                for k in range(len(points)):
                    if abs(p[0] - points[k][0]) < 1 and abs(p[1] - points[k][1]) < 1:
                        ind.append(k)
                        break
            if len(ind) == 3:
                delaunayTri.append(tuple(ind))
    return delaunayTri

def main():
    SAMPLE_FACE_PATH = 'elon.jpg'

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # Load sample face image
    sample_face = cv2.imread(SAMPLE_FACE_PATH)
    if sample_face is None:
        print(f"ERROR: Could not load sample face image from {SAMPLE_FACE_PATH}")
        sys.exit(1)

    # Get landmarks for sample face
    sample_points = extract_landmarks(sample_face, face_mesh)
    if sample_points is None:
        print("ERROR: Could not detect face landmarks in sample face image.")
        sys.exit(1)

    # Prepare image mask for sample face convex hull
    sample_mask = np.zeros(sample_face.shape[:2], dtype=np.uint8)
    sample_hull, sample_hull_indices = get_face_convex_hull(sample_points)
    cv2.fillConvexPoly(sample_mask, np.array(sample_hull, dtype=np.int32), 255)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    live_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    print("Press ESC or Q to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # Mirror image for selfie view
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = live_face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            cv2.putText(frame, "No Face Detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("ELON MUSK", frame)
            key = cv2.waitKey(1)
            if key in [27, ord('q'), ord('Q')]:
                break
            continue

        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        live_points = []
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            live_points.append((x, y))

        # Compute convex hull
        live_hull, live_hull_indices = get_face_convex_hull(live_points)

        # Delaunay triangulation for sample face
        rect = (0, 0, sample_face.shape[1], sample_face.shape[0])
        delaunay_tri = calculate_delaunay_triangles(rect, sample_points)

        # Warp sample_face triangles to live face
        warped_sample = np.zeros_like(frame)

        for tri_indices in delaunay_tri:
            t1 = [sample_points[i] for i in tri_indices]
            t2 = [live_points[i] for i in tri_indices]
            warp_triangle(sample_face, warped_sample, t1, t2)

        # Create mask for warped sample face
        warped_mask = np.zeros_like(frame[:, :, 0])
        cv2.fillConvexPoly(warped_mask, np.array(live_hull, dtype=np.int32), 255)

        center = (int(np.mean([p[0] for p in live_hull])), int(np.mean([p[1] for p in live_hull])))

        # Seamless clone to blend the warped face into the frame
        output = cv2.seamlessClone(warped_sample, frame, warped_mask, center, cv2.NORMAL_CLONE)

        cv2.putText(output, "ELON MUSK", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("ELON MUSK", output)

        key = cv2.waitKey(1)
        if key in [27, ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

