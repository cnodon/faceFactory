import cv2
import dlib
import os

def CreateFolder(path):

    del_path_space = path.strip()
    del_path_tail = del_path_space.rstrip('\\')

    isexists = os.path.exists(del_path_tail)

    if not isexists:
        os.makedirs(del_path_tail)
        return True
    else:
        return False

def DrawRectangle(frame, color, rect):
    cv2.rectangle(frame, (rect.left() - 10, rect.top() - 10), (rect.right() + 10, rect.bottom() + 10), color, 2)


def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):

    CreateFolder(path_name)
    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(camera_idx)

    detector = dlib.get_frontal_face_detector()

    color = (0, 255, 0)

    num = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print('not ok')
            break

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_dets = detector(img_gray, 1)
        det = None

        detected_len = len(face_dets)

        if detected_len == 0:
            continue
        elif detected_len > 1:
            # 只要最大的那个人脸
            temp_area = 0
            temp = 0
            for i, face_area in enumerate(face_dets):
                DrawRectangle(frame, color, face_area)
        else:
            det = face_dets[0]
            DrawRectangle(frame, color, det)


        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'num:%d' % (detected_len), (30, 30), font, 1, (255, 0, 255), 4)
        # 超过指定最大保存数量结束程序

        # 显示图像
        cv2.imshow(window_name, frame)
        # 按键盘‘Q’中断采集
        c = cv2.waitKey(25)
        if c & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        # 释放摄像头并销毁所有窗口


if __name__ == '__main__':

    new_user_name = 'PKR'
    window_name = 'Facial Recognition'  # 图像窗口
    camera_idx = 0  # 相机的ID号
    path = r'D:\image' + '/' + new_user_name
    CatchPICFromVideo(window_name, camera_idx, 10,  path)


