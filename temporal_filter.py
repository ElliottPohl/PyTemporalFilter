import cv2
import numpy as np
import argparse, os
from tqdm import tqdm

class TemporalFilter:
    def __init__(self, buffer_size=15, filter_type='max'):
        self.buffer_size = buffer_size
        self.filter_type = self.setFilter(filter_type)
        self.buffer_frames = []

    def update(self,img_input):
        self.buffer_frames.append(img_input)
        if len(self.buffer_frames) > self.buffer_size:
            del self.buffer_frames[0]

    def setFilter(self,filter_string):
        if filter_string.lower() == 'mean':
            return np.mean
        elif filter_string.lower() == 'median':
            return np.median
        elif filter_string.lower() == 'min':
            return np.min
        elif filter_string.lower() == 'max':
            return np.max

    def render(self):
        return np.array(self.filter_type(np.array(self.buffer_frames), axis=0), np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply a statistical function across a number of video frames')
    parser.add_argument('-i', '--input', required=True, help='Filepath for video input')
    parser.add_argument('-b', '--buffer_size', help='Amount of frames included in filter', default=15, type=int)
    parser.add_argument('-f', '--filter', help='Filter type', choices=('mean', 'median', 'min', 'max'), default='max')
    args = vars(parser.parse_args())

    video_path = os.path.abspath(args['input'])
    buffer_size = args['buffer_size']
    filter_type = args['filter']

    cap = cv2.VideoCapture(video_path)

    f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_fps = cap.get(cv2.CAP_PROP_FPS)
    f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    vid_fp_rt, vid_fp_ext = os.path.split(video_path)
    temp = vid_fp_ext.split('.')
    out_fn = f"{'.'.join(temp[:-1])}_{filter_type}_{buffer_size}.{temp[-1]}"
    out_fp = os.path.join(vid_fp_rt,out_fn)
    print(out_fp)

    out = cv2.VideoWriter(out_fp,f_fourcc,f_fps,(f_width,f_height))

    tf = TemporalFilter(buffer_size=buffer_size,filter_type=filter_type)

    for i in tqdm(range(f_count)):
        ret, frame = cap.read()
        if ret:
            tf.update(frame)
            img = tf.render()
            out.write(img)

    cap.release()
    out.release()




