import cv2
from brainframe.api import BrainFrameAPI
 
 
def read_frame(stream_uri, frame_index):
    cap = cv2.VideoCapture(stream_uri)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    rst, frame = cap.read()
    if not rst:
        print(f"Failed to read frame: {frame_index}")
    cap.release()
    return frame
 
 
def detect_image(api, frame, capsule_names=None):
    if capsule_names is None:
        capsule_names = ["detector_person_and_vehicle_fast", "detector_face_fast"]
    detections = api.process_image(frame, capsule_names, {})
    return detections
 
 
def main():
    # The capsules for person and face detection
    capsule_names = ["detector_person_and_vehicle_fast", "detector_face_fast"]
 
    # The video file name, it can be replaced by the other video file or rtsp/http streams
    stream_path = "Lesson 11 - Summary.mp4"
 
    # The url to access the brainframe server with rest api
    bf_server_url = "http://localhost"
 
    api = BrainFrameAPI(bf_server_url)
    print(api)
    api.wait_for_server_initialization()
    
    frame = read_frame(stream_path, 5)
    if frame is None:
        return
 
    detections = detect_image(api, frame, capsule_names)
    print(detections)
 
    # Could extend the feature here to render and crop persons
 
 
if __name__ == "__main__":
    main()