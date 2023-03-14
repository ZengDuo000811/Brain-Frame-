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
 
def capture_video(stream_path, coords):
    cap = cv2.VideoCapture(stream_path)

    # Define the coordinates of the area to crop(定义要裁剪的区域的坐标)
    x, y, w, h = coords[0][0], coords[0][1], coords[1][0] - coords[0][0], coords[2][1] - coords[1][1]

    # Create a VideoWriter object to write video frames(创建一个VideoWriter对象来写入视频帧)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (w,h))

    frame_count = 0
    # Loop read video frames(循环读取视频帧)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop frame(裁剪帧)
        cropped_frame = frame[y:y+h, x:x+w]

        # Write the cropped frame to the output file(将裁剪后的帧写入输出文件)
        out.write(cropped_frame)

        frame_count += 1
        cv2.imwrite("./img/" + 'frame' + str(frame_count) + '.jpg', frame[y:y+h, x:x+w])
        # Show cropped frames(显示裁剪后的帧)
        cv2.imshow('frame', cropped_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear(清理)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return
 
def main():
    # The capsules for person and face detection
    capsule_names = ["detector_person_and_vehicle_fast", "detector_face_fast"]
 
    # The video file name, it can be replaced by the other video file or rtsp/http streams
    stream_path = "./Lesson 11 - Summary.mp4"
 
    # The url to access the brainframe server with rest api
    bf_server_url = "http://localhost"
 
    api = BrainFrameAPI(bf_server_url)
    # print(api)
    api.wait_for_server_initialization()
    
    frame = read_frame(stream_path, 5)
    if frame is None:
        return
 
    detections = detect_image(api, frame, capsule_names)
    # print(detections[0].coords)

 
    # Could extend the feature here to render and crop persons
    capture_video(stream_path, detections[0].coords)
 
 
if __name__ == "__main__":
    main()