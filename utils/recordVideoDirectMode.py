import cv2
import os
def record_direct_mode(p,step,img_dir='./images'):
    # Set up the camera
    width = 640
    height = 480
    fov = 60
    aspect = width / height
    near = 0.01
    far = 100
    camera_pos = [-2, 0, 2]
    target_pos = [0, 0, 1]
    up_vector = [1, 0, 0]
    view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get the camera image
    _, _, img, _, _ = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix, shadow=False, lightDirection=[1, 1, 1], renderer=p.ER_TINY_RENDERER) #renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # Save the image
    cv2.imwrite(os.path.join(img_dir,str(step) + ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def record_gui_mode(p,step,img_dir='./images'):
    # img = p.getCameraImage(224, 224, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    # Compute the camera view matrix
    info = p.getDebugVisualizerCamera()
    # print(info)
    viewMatrix = info[2]
    projectionMatrix = info[3]
    # Get the camera image
    _, _, img, depth, segm = p.getCameraImage(
        width=640, #1280, #640,
        height=480, #720, #480,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        shadow=False,
        lightDirection=[1, 1, 1],
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    # print(img)
    # Save the image
    cv2.imwrite(os.path.join(img_dir,str(step) + ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def stitch_video_direct_mode(episode,img_dir='./images'):
    # Read the saved PNG images and combine them into a video file
    width = 640
    height = 480
    out = cv2.VideoWriter(f"./results/video_ep_{episode}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))

    file_list = os.listdir(img_dir)
    file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
    for file in file_list:
        img = cv2.imread(os.path.join(img_dir, file))
        out.write(img)
    out.release()

    # Delete each file in the directory
    for file in file_list:
        file_path = os.path.join(img_dir, file)
        os.remove(file_path)