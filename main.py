import cv2
import numpy as np

#INITIALISION
np.random.seed(19)

#CLASSES
class TrackedPoint:
    def __init__(self, pos, life=100, box_dim=40):
        self.pos = np.array(pos, dtype=np.float32)  # x, y coordinates
        self.life = life  # frames remaining
        self.box_dim = box_dim

#HELPER FUNCTIONS
def initialise(src=0, nfeatures=500, num_points=5):
    """
    Initialise video capture, ORB detector, and Lucas-Kanade optical flow for tracking.

    Parameters
    ----------
    src : int or str, optional
        Video source (default is 0, first webcam). Can be a camera index or video file path.
    nfeatures : int, optional
        Maximum number of ORB features to detect (default 500).
    num_points : int, optional
        Number of strongest keypoints to track.

    Returns
    -------
    cap : cv2.VideoCapture
        Video capture object.
    orb : cv2.ORB
        ORB feature detector.
    lk_params : dict
        Lucas-Kanade optical flow parameters.
    prev_grey : np.ndarray
        Grayscale first frame.
    initial_kps : list of cv2.KeyPoint
        List of strongest ORB keypoints.
    prev_pts : np.ndarray
        Initial points for LK optical flow, shape (-1,1,2).
    """

    cap = cv2.VideoCapture(src)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    lk_params = dict(
        winSize=(21,21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise SystemExit("Cannot open camera")

    prev_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints
    kps = orb.detect(prev_grey, None)
    if not kps:
        cap.release()
        raise SystemExit("No keypoints detected")

    # Sort by response (strength) and take top `num_points`
    initial_kps = sorted(kps, key=lambda k: k.response, reverse=True)[:num_points]

    # Convert to NumPy array for LK optical flow
    prev_pts = np.array([kp.pt for kp in initial_kps], dtype=np.float32).reshape(-1,1,2)

    return cap, orb, lk_params, prev_grey, initial_kps, prev_pts

def select_initial_point(grey, orb):
    '''
    Detect the intial keypoint using the result from the orb feature detector.

    Parameters
    ----------
    grey: greyscale frame
    orb: cv2.ORB
        ORB feature detector

    finish commenting later
    '''
    initial_kps = orb.detect(grey, None)
    if not initial_kps:
        raise SystemExit("No keypoints detected")
    initial_kps = sorted(initial_kps, key=lambda k: k.response, reverse=True)
    x, y = initial_kps[0].pt
    return np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)


def track_loop(cap, orb, lk_params, prev_grey, initial_kps, num_points=5, box_dim=40, jitter=5, neighbors=2):
    """
    Track multiple ORB keypoints with Lucas-Kanade optical flow,
    draw jittered rectangles, and connect nearest neighbors with lines.

    Parameters
    ----------
    cap : cv2.VideoCapture
        OpenCV video capture object.
    orb : cv2.ORB
        ORB feature detector.
    lk_params : dict
        Lucas-Kanade optical flow parameters.
    prev_grey : np.ndarray
        Initial grayscale frame.
    initial_kps : list
        Initial ORB keypoints to track.
    num_points : int
        Maximum number of points to track.
    box_dim : int
        Side length of rectangles.
    jitter : int
        Maximum pixel jitter for boxes.
    neighbors : int
        Number of nearest neighbors to connect lines.
    """

    # Initialize tracked points with the strongest keypoints
    tracked_points = [TrackedPoint(kp.pt, life=200, box_dim=box_dim) for kp in sorted(initial_kps, key=lambda k: k.response, reverse=True)[:num_points]]
    print(tracked_points)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track existing points
        if tracked_points:
            prev_pts = np.array([p.pos for p in tracked_points], dtype=np.float32).reshape(-1,1,2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_grey, grey, prev_pts, None, **lk_params)

            new_tracked = []
            for tp, np_pt, st in zip(tracked_points, next_pts.reshape(-1,2), status.reshape(-1)):
                if st == 1:  # successfully tracked
                    tp.pos = np_pt
                    tp.life -= 1
                    if tp.life > 0:
                        new_tracked.append(tp)
            tracked_points = new_tracked

        # Spawn new points if less than num_points
        if len(tracked_points) < num_points:
            kps = orb.detect(grey, None)
            if kps:
                kps = sorted(kps, key=lambda k: k.response, reverse=True)
                for kp in kps[:num_points - len(tracked_points)]:
                    tracked_points.append(TrackedPoint(kp.pt, life=200, box_dim=box_dim))

        coords = np.array([p.pos for p in tracked_points], dtype=np.float32)

        # Draw lines to nearest neighbors
        for i, p in enumerate(coords):
            if len(coords) > 1:
                dists = np.linalg.norm(coords - p, axis=1)
                nearest_idx = np.argsort(dists)[1:neighbors+1]
                for j in nearest_idx:
                    cv2.line(frame, tuple(p.astype(int)), tuple(coords[j].astype(int)), (255,255,255), 1)

        # Draw jittered boxes and labels
        for tp in tracked_points:
            x, y = tp.pos
            jx = np.random.randint(-jitter, jitter+1)
            jy = np.random.randint(-jitter, jitter+1)

            tl = (int(x - tp.box_dim//2) + jx, int(y - tp.box_dim//2) + jy)
            br = (int(x + tp.box_dim//2) + jx, int(y + tp.box_dim//2) + jy)

            # Clip to frame
            tl = (max(tl[0],0), max(tl[1],0))
            br = (min(br[0], frame.shape[1]-1), min(br[1], frame.shape[0]-1))


            # Draw rectangle
            cv2.rectangle(frame, tl, br, (255,255,255), 2)

            # Optional: draw coordinates as label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Pick contrasting color based on box brightness
            roi = frame[tl[1]:br[1], tl[0]:br[0]]
            if roi.size:
                text_content = np.random.rand() < 0.5
                if text_content:
                    text = f"({int(x)},{int(y)})"
                else:
                    avg_color = np.mean(roi, axis=(0,1))
                    avg_color_int = tuple(int(np.clip(c, 0, 255)) for c in avg_color)
                    # avg_color is BGR; convert to RGB hex
                    text = "#{:02x}{:02x}{:02x}".format(avg_color_int[2], avg_color_int[1], avg_color_int[0])
                inverted_roi = cv2.bitwise_not(roi)
                frame[tl[1]:br[1], tl[0]:br[0]] = inverted_roi
                mean_brightness = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean()
                color = (0,0,0) if mean_brightness > 127 else (255,255,255)
            else:
                color = (0,0,0)
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            text_x = tl[0] + max((br[0] - tl[0] - text_w)//2, 0)
            text_y = tl[1] + (br[1] - tl[1] + text_h)//2
            text_y = max(text_y, tl[1]+text_h)
            text_y = min(text_y, br[1]-baseline)
            text_x = min(text_x, br[0]-text_w)

            # Draw text with outline
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Show frame
        cv2.imshow("multi-track", frame)
        prev_grey = grey.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    cap = None
    try:
        cap, orb, lk_params, prev_grey, initial_kps, prev_pts = initialise()
        track_loop(cap, orb, lk_params, prev_grey, initial_kps)  # Use initial_kps not prev_pts
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
