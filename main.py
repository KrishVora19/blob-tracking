import cv2
import numpy as np
from typing import Tuple

#INITIALISION
np.random.seed(19)

#CLASSES
class TrackedPoint:
    def __init__(
            self,
            pos,
            life=100,
            box_dim=40
            ):
        
        self.pos = np.array(pos, dtype=np.float32)  # x, y coordinates
        self.life = life  # frames remaining
        self.box_dim = box_dim

class LabelStyle:
    def __init__(
            self,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            scale:float=0.5,
            thickness:float=1,
            color: Tuple[int, int, int]=(255,255,255)
            ):
        self.font = font
        self.font_scale = scale
        self.font_thickness = thickness
        self.color = color

    def draw(self, frame, text, pos):
        cv2.putText(
            frame,
            text,
            pos,
            self.font,
            self.font_scale,
            self.color,
            self.font_thickness,
            cv2.LINE_AA
        )
    
    def get_size(self,text):
        (w,h), baseline = cv2.getTextSize(
            text,
            self.font,
            self.font_scale,
            self.font_thickness
        )
        return (w,h),baseline
    

#HELPER FUNCTIONS
def initialise(src=0, nfeatures=5000, num_points=10):
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

def back_off(new_pt, tracked_points, min_dist=100):
    for tp in tracked_points:
        if np.linalg.norm(new_pt - tp.pos) < min_dist:
            return False
    return True

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

def gameboy_greenscale(roi):

    # Luminance
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 4-shade palette
    levels = 4
    quantized = np.floor(gray / (256 / levels)) * (256 / (levels - 1))
    quantized = quantized.astype(np.uint8)

    # Mapping grey to green
    palette = np.array([
        [15, 56, 15],    # Dark green
        [48, 98, 48],    # Medium dark
        [139, 172, 15],  # Medium light
        [155, 188, 15]   # Light green
    ], dtype=np.uint8)
    indices = (quantized / (256 / (levels - 1))).astype(int)
    green_img = palette[indices]

    return green_img

def draw_motion_vectors(frame, prev_pts, next_pts, color=(0,0,255)):
    for (x1,y1),(x2,y2) in zip(prev_pts.reshape(-1,2),next_pts.reshape(-1,2)):
        starting_point = (int(x1),int(y1))
        ending_point = (int(x2),int(y2))

        cv2.arrowedLine(frame, starting_point, ending_point, color, thickness=1, tipLength=1)

def track_loop(cap, orb, lk_params, prev_grey, initial_kps, num_points=10, box_dim=40, jitter=1, box_fluctuation=0.05, neighbors=2):
    # Initialize
    tracked_points = [TrackedPoint(kp.pt, life=100, box_dim=box_dim)
                    for kp in sorted(initial_kps, key=lambda k: k.response, reverse=True)[:num_points]]
    style = LabelStyle(scale=0.5,thickness=1,color=(255,255,255))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        prev_pts = None
        next_pts = None
        status = None

        # Track existing points if any
        if tracked_points:
            prev_pts = np.array([p.pos for p in tracked_points], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_grey, grey, prev_pts, None, **lk_params)

            # If LK returned something, filter for successful tracks
            if status is not None:
                st = status.reshape(-1)
                # update tracked_points list using status
                new_tracked = []
                for tp, np_pt, s in zip(tracked_points, next_pts.reshape(-1, 2), st):
                    if s == 1:
                        tp.pos = np_pt
                        tp.life -= 1
                        if tp.life > 0:
                            new_tracked.append(tp)
                tracked_points = new_tracked
            else:
                # no valid status -> clear tracked points
                tracked_points = []

        # Draw motion vectors only if we have valid prev/next and status
        if prev_pts is not None and next_pts is not None and status is not None:
            good_mask = (status.reshape(-1) == 1)
            if good_mask.any():
                good_prev = prev_pts.reshape(-1, 2)[good_mask]    # shape (M,2)
                good_next = next_pts.reshape(-1, 2)[good_mask]    # shape (M,2)
                # pass arrays shaped (M,2) or (M,1,2) both supported
                draw_motion_vectors(frame, good_prev, good_next)

        # Spawn new points if needed
        if len(tracked_points) < num_points:
            kps = orb.detect(grey, None)
            if kps:
                kps = sorted(kps, key=lambda k: k.response, reverse=True)
                for kp in kps:
                    if len(tracked_points) >= num_points:
                        break
                    pt = np.array(kp.pt)
                    if back_off(pt, tracked_points):
                        tracked_points.append(TrackedPoint(kp.pt, life=200, box_dim=box_dim))

        coords = np.array([p.pos for p in tracked_points], dtype=np.float32) if tracked_points else np.empty((0,2), dtype=np.float32)

        # Draw lines to nearest neighbors
        if len(coords) > 1:
            for i, p in enumerate(coords):
                dists = np.linalg.norm(coords - p, axis=1)
                nearest_idx = np.argsort(dists)[1:neighbors+1]

                
                for j in nearest_idx:
                    p1 = tuple(p.astype(int))
                    p2 = tuple(coords[j].astype(int))
                    cv2.line(frame, p1, p2, (255,255,255), 1)
                    mid_x = int((p1[0] + p2[0])/2)
                    mid_y = int((p1[1] + p2[1])/2)
                    style.draw(frame,(str(dists[j])),(mid_x,mid_y))

        for tp in tracked_points:
            x, y = tp.pos
            jx = np.random.randint(-jitter, jitter+1)
            jy = np.random.randint(-jitter, jitter+1)

            x_multiplier = np.random.uniform(1-box_fluctuation, 1+box_fluctuation)
            y_multiplier = np.random.uniform(1-box_fluctuation, 1+box_fluctuation)

            tl = (int((x - tp.box_dim//2)*x_multiplier) + jx, int((y - tp.box_dim//2)*y_multiplier) + jy)
            br = (int((x + tp.box_dim//2)*x_multiplier) + jx, int((y + tp.box_dim//2)*y_multiplier) + jy)

            # Clip to frame and ensure ordering
            tl = (max(0, min(tl[0], br[0])), max(0, min(tl[1], br[1])))
            br = (max(tl[0], min(br[0], frame.shape[1]-1)), max(tl[1], min(br[1], frame.shape[0]-1)))

            roi = frame[tl[1]:br[1], tl[0]:br[0]]
            if roi.size:
                if np.random.rand() < 0.3:
                    frame[tl[1]:br[1], tl[0]:br[0]] = cv2.bitwise_not(roi)
                else:
                    frame[tl[1]:br[1], tl[0]:br[0]] = gameboy_greenscale(roi)

            cv2.rectangle(frame, tl, br, (255,255,255), 2)

            # label
            text = f"({int(x)},{int(y)})"
            (text_w, text_h), baseline = style.get_size(text)
            text_x = tl[0] + max((br[0] - tl[0] - text_w)//2, 0)
            text_y = tl[1] + (br[1] - tl[1] + text_h)//2
            text_y = max(text_y, tl[1]+text_h)
            text_y = min(text_y, br[1]-baseline)
            text_x = min(text_x, br[0]-text_w)

            # contrast color
            if roi.size:
                mean_brightness = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean()
                color = (0,0,0) if mean_brightness > 127 else (255,255,255)
            else:
                color = (0,0,0)
            style.draw(frame,text,(text_x,text_y))

        # Show frame
        cv2.imshow("multi-track", frame)
        prev_grey = grey.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#MAIN LOOP
def main():
    cap = None
    try:
        cap, orb, lk_params, prev_grey, initial_kps, prev_pts = initialise()
        track_loop(cap, orb, lk_params, prev_grey, initial_kps)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
