import cv2
import numpy as np
import argparse, os
from tqdm import tqdm

class TemporalFilter:
    #TODO: Try variations using different color spaces e.g. HSV or YUV and mixing function use per channel
    #TODO: Add ability to widen amount of frames in the center of the harris filter, so the effect is only visible
    #on the edges of movement.

    #TODO: add slitscan module
    def __init__(self, buffer_size=15, filter_type='max', harris_filter=0.0, harris_reverse=False,
                 harris_slide=False):
        self.buffer_size = buffer_size
        self.filter_type = self.setFilter(filter_type)
        self.buffer_frames = []
        self.harris_filter = harris_filter
        self.harris_reverse = harris_reverse
        self.harris_slide = harris_slide

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

    def indexClamp(self,index_val,buffer_len):
        if index_val < 1:
            return 1
        elif index_val > buffer_len:
            return buffer_len
        else:
            return index_val

    def harrisPoolingIndexes(self):
        buffer_len = len(self.buffer_frames)
        # Amount of frames to remove from full buffer
        frame_split = int(buffer_len / 3)

        f_max = buffer_len
        f_mid = self.indexClamp((f_max - frame_split), buffer_len)
        f_min = self.indexClamp((f_mid - frame_split), buffer_len)

        if self.harris_reverse:
            r_index = ((buffer_len - f_max), buffer_len)
            g_index = ((buffer_len - f_mid), buffer_len)
            b_index = ((buffer_len - f_min), buffer_len)

            return (b_index, g_index, r_index)
        else:
            r_index = ((buffer_len - f_min), buffer_len)
            g_index = ((buffer_len - f_mid), buffer_len)
            b_index = ((buffer_len - f_max), buffer_len)

            return (b_index, g_index, r_index)

    def harrisSlidingIndexes(self):
        buffer_len = len(self.buffer_frames)
        # Amount of frames to remove from full buffer
        frame_split = int((buffer_len/2)+1)
        gap = buffer_len - frame_split

        if self.harris_reverse:
            r_index = (0, frame_split)
            g_index = (int(gap / 2), int(buffer_len - (gap / 2)))
            b_index = (gap, buffer_len)

            return (b_index, g_index, r_index)
        else:
            r_index = (gap,buffer_len)
            g_index = (int(gap/2),int(buffer_len-(gap/2)))
            b_index = (0, frame_split)

            return (b_index, g_index, r_index)

    def simple_frame(self,frame_arr):
        return np.array(self.filter_type(frame_arr, axis=0), np.uint8)

    def harris_frame(self,frame_arr):
        if self.harris_slide:
            idx = self.harrisSlidingIndexes()
        else:
            idx = self.harrisPoolingIndexes()

        if len(frame_arr[0].shape) == 3:
            r = np.array(self.filter_type(frame_arr[idx[2][0]:idx[2][1], :, :, 2], axis=0), np.uint8)
            g = np.array(self.filter_type(frame_arr[idx[1][0]:idx[1][1], :, :, 1], axis=0), np.uint8)
            b = np.array(self.filter_type(frame_arr[idx[0][0]:idx[0][1], :, :, 0], axis=0), np.uint8)
            bgr = cv2.merge((b,g,r))
            return bgr
        else:
            r = np.array(self.filter_type(frame_arr[idx[2][0]:idx[2][1], :, :], axis=0), np.uint8)
            g = np.array(self.filter_type(frame_arr[idx[1][0]:idx[1][1], :, :], axis=0), np.uint8)
            b = np.array(self.filter_type(frame_arr[idx[0][0]:idx[0][1], :, :], axis=0), np.uint8)
            bgr = cv2.merge((b, g, r))
            return bgr

    def render(self):
        frame_arr = np.array(self.buffer_frames)

        if self.harris_filter:
            return self.harris_frame(frame_arr)
        else:
            return self.simple_frame(frame_arr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply a statistical function across a number of video frames')
    parser.add_argument('-i', '--input', required=True, help='Filepath for video input')
    parser.add_argument('-o', '--output', help='Filepath for video output')
    parser.add_argument('-b', '--buffer_size', help='Amount of frames included in filter', default=15, type=int)
    parser.add_argument('-f', '--filter', help='Filter type', choices=('mean', 'median', 'min', 'max'), default='max')
    parser.add_argument('-g','--grayscale',help='Convert frames to grayscale before adding frame to buffer.',
                        action='store_true')
    parser.add_argument('-r', '--resize', help='resize ratio', default=1.0, type=float)
    parser.add_argument('-hf','--harris_filter',help='split buffer selection over color channels',action='store_true')
    parser.add_argument('-rh', '--reverse_harris', help='Reverses harris order, RGB instead of BGR',
                        action='store_true')
    parser.add_argument('-hs', '--harris_slide',
                        help='Uses a shrinking pool of frames instead of a sliding slice of frames'
                             ' to generate color shift',
                        action='store_false')
    parser.add_argument('-s', '--speed',
                        help='Default is 1 frame in 1 frame out, you can specify 2 frames in or higher',
                        default=1, type=int)
    parser.add_argument('-fps', '--frames_per_second',
                        help='Default is 30 if neither the user or video file specifies',
                        default=999, type=int)
    parser.add_argument('-loop', '--loop_preload',
                        help='preloads frame buffer with frames from end of video',
                        action='store_true')

    args = vars(parser.parse_args())

    video_path = os.path.abspath(args['input'])
    output_filename = args['output']
    buffer_size = args['buffer_size']
    filter_type = args['filter']
    grayscale = args['grayscale']
    harris_filter = args['harris_filter']
    harris_reverse = args['reverse_harris']
    harris_slide = args['harris_slide']
    resize_amt = args['resize']
    speed_up = args['speed']
    looped = args['loop_preload']

    cap = cv2.VideoCapture(video_path)

    f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_fps = cap.get(cv2.CAP_PROP_FPS)
    f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*resize_amt)
    f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*resize_amt)
    f_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    f_fps = f_fps if args['frames_per_second'] == 999 else args['frames_per_second']

    if output_filename == None:
        vid_fp_rt, vid_fp_ext = os.path.split(video_path)
        temp = vid_fp_ext.split('.')
        out_fn = f"{'.'.join(temp[:-1])}_F-{filter_type}_B-{buffer_size}" \
                 f"{'_HARRIS' if harris_filter else ''}{'_Gray' if grayscale else ''}{'_RH' if harris_reverse else ''}" \
                 f"{'' if harris_slide else '_HS'}{'' if speed_up <= 1 else '_X'+str(speed_up)}.{temp[-1]}"
        out_fp = os.path.join(vid_fp_rt,out_fn)
    else:
        out_fp = os.path.abspath(output_filename)

    print(out_fp)
    out = cv2.VideoWriter(out_fp,f_fourcc,f_fps,(f_width,f_height))

    tf = TemporalFilter(buffer_size=buffer_size, filter_type=filter_type,
                        harris_filter=harris_filter, harris_reverse=harris_reverse, harris_slide=harris_slide)

    # This does the same thing as below, but it loads the end of the clip into the buffer to make a smoother loop
    if looped:
        preload_start = f_count - buffer_size
        cap.set(cv2.CAP_PROP_POS_FRAMES,preload_start)
        print(f"preloading {buffer_size} frames from end of video")

        for i in range(buffer_size):
            ret, frame = cap.read()
            if ret:
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if resize_amt != 1.0:
                    frame = cv2.resize(frame, (0, 0), fx=resize_amt, fy=resize_amt, interpolation=cv2.INTER_CUBIC)
                tf.update(frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    for i in tqdm(range(f_count)):
        ret, frame = cap.read()
        if ret:
            if grayscale:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if resize_amt != 1.0:
                frame = cv2.resize(frame,(0,0),fx=resize_amt,fy=resize_amt,interpolation=cv2.INTER_CUBIC)
            tf.update(frame)

            if i % speed_up == 0:
                img = tf.render()
                out.write(img)

    cap.release()
    out.release()

