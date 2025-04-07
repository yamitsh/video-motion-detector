import multiprocessing as mp
import cv2
from datetime import datetime
import argparse
import imutils
import time


def streamer(video_path, detector_queue):
    """
    reads a video file frame by frame and sends each frame to the detector
    """
    cap = cv2.VideoCapture(video_path)

    while True:
        # grab the current frame
        ret, frame = cap.read()
        if not ret:
            print("streamer finished")
            break
        # put the frame in the queue
        detector_queue.put(frame)
        time.sleep(0.03)

    # the video is over, stop the detector process
    detector_queue.put(None)
    cap.release()


def detector(detector_queue, displayer_queue):
    """
    receives frames from the streamer, performs motion detection,
    and sends the original frame and detection information to the displayer
    """

    counter = 0
    prev_frame = None

    while True:
        frame = detector_queue.get()
        if frame is None:
            # stop the displayer
            displayer_queue.put(None)
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if counter == 0:
            prev_frame = gray_frame
            counter += 1
        else:
            diff = cv2.absdiff(gray_frame, prev_frame)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            prev_frame = gray_frame
            counter += 1

            detections = []
            for contour in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append((x, y, w, h))

            displayer_queue.put((frame, detections))
            time.sleep(0.01)

    print("detector finished")


def displayer(displayer_queue):
    """
    receives frames and detection information, draws detections and timestamp,
    and displays the video
    """
    while True:
        item = displayer_queue.get()
        if item is None:
            break

        frame, detections = item

        # draw detections (rectangles)
        for x, y, w, h in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # write current time
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, now, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # show the frame and record if the user presses a key
        cv2.imshow("Video with Detections", frame)

        # if the `q` key is pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("displayer finished")


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    args = vars(ap.parse_args())

    video_path = args["video"] or "People_Video.mp4"

    # create communication queues
    detector_queue = mp.Queue(maxsize=30)
    displayer_queue = mp.Queue(maxsize=30)

    # create processes
    streamer_p = mp.Process(target=streamer, args=(video_path, detector_queue))
    detector_p = mp.Process(target=detector, args=(detector_queue, displayer_queue))
    displayer_p = mp.Process(target=displayer, args=(displayer_queue,))

    # start the processes
    streamer_p.start()
    detector_p.start()
    displayer_p.start()
    