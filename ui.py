import cv2

# =========================
# #1 UI Config
# =========================
WINDOW_NAME = "Pedestrian Counter"
FRAME_WIDTH = 960


# =========================
# #2 UI Init
# =========================
def init_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, int(FRAME_WIDTH * 9 / 16))
    cv2.waitKey(1)


# =========================
# #3 Loading Screen
# =========================
def show_loading(frame, scale):
    frame_show = cv2.resize(frame, None, fx=scale, fy=scale)
    h, w = frame_show.shape[:2]

    cv2.putText(
        frame_show,
        "Now loading...",
        (w // 2 - 120, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )
    cv2.imshow(WINDOW_NAME, frame_show)
    cv2.waitKey(1)


# =========================
# #4 Counter Overlay
# =========================
def draw_counter(frame, in_count, out_count):
    total = in_count + out_count

    cv2.rectangle(frame, (10, 10), (260, 120), (0, 0, 0), -1)

    cv2.putText(frame, f"IN    : {in_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT   : {out_count}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"TOTAL : {total}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


# =========================
# #5 Show Frame
# =========================
def show_frame(frame, scale):
    frame_show = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2.imshow(WINDOW_NAME, frame_show)
