import cv2
import numpy as np

class VideoBuffer:
    def __init__(self,max_size=None):
        self.buffer = []
        self.buffer_max = max_size

    def update(self, frame):
        self.buffer.append(frame)
        if len(self.buffer) > self.buffer_max:
            del self.buffer[0]

    def getBuffer(self):
        return self.buffer

    def getArray(self):
        return np.array(self.buffer)

def pixelSort(frame,vert=False):
    if vert:
        return np.sort(frame,1)
    else:
        return np.sort(frame, 0)

class MotionDetection:
    def __init__(self,shadows=False):
        self.bg_det = cv2.createBackgroundSubtractorKNN(detectShadows=shadows)
        self.canv = None
        self.frame = None
        self.seg = None

    def update(self,frame):
        seg = self.bg_det.apply(frame)

        if type(self.canv) == type(None):
            self.canv = frame
            self.seg = seg
            self.frame = frame
        else:
            self.seg = seg
            self.frame = frame
            fg = cv2.bitwise_or(frame, frame, mask=seg)
            msk_inv = cv2.bitwise_not(seg)
            bg = cv2.bitwise_or(self.canv, self.canv, mask=msk_inv)
            self.canv = cv2.bitwise_or(fg,bg)

    def render(self):
        fg = cv2.bitwise_or(self.frame, self.frame, mask=self.seg)
        msk_inv = cv2.bitwise_not(self.seg)
        bg = cv2.bitwise_or(self.canv, self.canv, mask=msk_inv)
        self.canv = cv2.bitwise_or(fg, bg)
        return self.canv

class BeltScan:
    # slit scan but slits get pushed off at the end
    def __init__(self,pixels=20,vert=False):
        self.bg = None
        self.pixels = pixels
        self.vert = vert

    def update(self,frame):
        if type(self.bg) == type(None):
            self.bg = frame
        else:
            if self.vert:
                self.bg = np.concatenate((self.bg[self.pixels:,:,:],frame[:self.pixels,:,:]),)
            else:
                self.bg = np.concatenate((self.bg[:,self.pixels:,:],frame[:,:self.pixels,:]),1)

    def render(self):
        return self.bg

class SlitScan:
    # bg image
    # this filter works by shifting slice values
    # we maintain a background image and use this as a place to paste values
    # we want the background to stay the same but have the LIVE portion of it get smaller
    # TODO: add dimension check. If video has more frame shift than size will throw error
    def __init__(self,pixels=1,vert=False):
        self.vert = vert
        self.pixels = pixels
        self.position = 1
        self.bg = None

    def update(self,frame):
        if type(self.bg) == type(None):
            self.bg = frame
        else:
            if self.vert:
                self.bg = np.concatenate((self.bg[:self.position,:,:],frame[self.position:,:,:]))
            else:
                self.bg = np.concatenate((self.bg[:, :self.position, :], frame[:, self.position:, :]),1)

            self.position += self.pixels

    def render(self):
        return self.bg

class TemporalFilter:
    def __init__(self, buffer_size=15, filter_type='max'):
        self.vb = VideoBuffer(buffer_size)
        self.filter_type = self.setFilter(filter_type)

    def update(self, img_input):
        self.vb.update(img_input)

    def setFilter(self, filter_string):
        if filter_string.lower() == 'mean':
            return np.mean
        elif filter_string.lower() == 'median':
            return np.median
        elif filter_string.lower() == 'min':
            return np.min
        elif filter_string.lower() == 'max':
            return np.max

    def render(self):
        return np.array(self.filter_type(self.vb.getBuffer(), axis=0), np.uint8)

class TemporalBuildUp:
    def __init__(self, filter_type='max'):
        self.bg = None
        self.filter_type = self.setFilter(filter_type)

    def setFilter(self, filter_string):
        if filter_string.lower() == 'min':
            return np.min
        else:
            return np.max

    def update(self, frame):
        if type(self.bg) == type(None):
            self.bg = frame
        else:
            self.bg = np.array(self.filter_type(np.array([self.bg,frame]), axis=0), np.uint8)



    def render(self):
        return self.bg

class HarrisSimple:
    def __init__(self,reverse=False):
        self.vb = VideoBuffer(max_size=3)
        self.reverse = reverse

    def update(self, frame):
        self.vb.update(frame)

    def render(self):
        if len(self.vb.buffer) >= 3:
            if self.reverse:
                b = self.vb.buffer[0][:, :, 0]
                g = self.vb.buffer[1][:, :, 1]
                r = self.vb.buffer[2][:, :, 2]
                return cv2.merge((b,g,r))
            else:
                b = self.vb.buffer[2][:, :, 0]
                g = self.vb.buffer[1][:, :, 1]
                r = self.vb.buffer[0][:, :, 2]
                return cv2.merge((b, g, r))
        else:
            return self.vb.buffer[0]

class HarrisTemporal:
    def __init__(self, buffer_size=15, filter_type='max', align='overlap',reverse=False):
        # align options right, left, center, overlap
        self.vb = VideoBuffer(buffer_size)
        self.reverse = reverse
        self.filt = self.setFilter(filter_type)
        self.idx = self.setAlign(align)

    def update(self, img_input):
        self.vb.update(img_input)

    def setFilter(self, filter_string):
        if filter_string.lower() == 'mean':
            return np.mean
        elif filter_string.lower() == 'median':
            return np.median
        elif filter_string.lower() == 'min':
            return np.min
        elif filter_string.lower() == 'max':
            return np.max

    def setAlign(self, filter_string):
        if filter_string.lower() == 'right':
            return self.right
        elif filter_string.lower() == 'left':
            return self.left
        elif filter_string.lower() == 'center':
            return self.center
        elif filter_string.lower() == 'overlap':
            return self.overlap
        elif filter_string.lower() == 'overlap_rev':
            return self.overlap_rev

    # this is an out of range error check
    def indexClampEnd(self, index_val, buffer_len):
        # checks if value is in a valid range
        if index_val < 1:
            return 1
        elif index_val > buffer_len:
            return buffer_len
        else:
            return index_val

    def right(self):
        # ((0, 5), (0, 10), (0, 15))
        ##     red
        ####   green
        ###### blue
        # ((0, 15), (0, 10), (0, 5))
        ###### red
        ####   green
        ##     blue

        buffer_len = len(self.vb.buffer)
        # Amount of frames to remove from full buffer
        frame_split = int(buffer_len / 3)

        f_max = buffer_len
        f_mid = self.indexClampEnd((f_max - frame_split), buffer_len)
        f_min = self.indexClampEnd((f_mid - frame_split), buffer_len)

        if self.reverse:
            r_index = (0, f_min)
            g_index = (0, f_mid)
            b_index = (0, f_max)
            return (b_index, g_index, r_index)
        else:
            # ((0, 15), (0, 10), (0, 5))
            r_index = (0, f_max)
            g_index = (0, f_mid)
            b_index = (0, f_min)
            return (b_index, g_index, r_index)

    def left(self):
            ## red
          #### green
        ###### blue
        # ((10, 15), (5, 15), (0, 15))
        ###### red
        ####   green
        ##     blue
        # ((0, 15), (5, 15), (10, 15))


        buffer_len = len(self.vb.buffer)
        frame_split = int(buffer_len / 3)

        f_max = buffer_len
        f_mid = self.indexClampEnd((f_max - frame_split), buffer_len)
        f_min = self.indexClampEnd((f_mid - frame_split), buffer_len)

        if self.reverse:
            r_index = ((buffer_len - f_max), buffer_len)
            g_index = ((buffer_len - f_mid), buffer_len)
            b_index = ((buffer_len - f_min), buffer_len)

            return (b_index, g_index, r_index)
        else:
            r_index = ((buffer_len - f_min), buffer_len)
            g_index = ((buffer_len - f_mid), buffer_len)
            b_index = ((buffer_len - f_max), buffer_len)

            return (b_index, g_index, r_index)

    def center(self):
        ###### red
         ####  green
          ##   blue
          ##   red
         ####  green
        ###### blue

        end = len(self.vb.getBuffer())
        fs = int((end / 3))

        if self.reverse:
            r_s = fs
            r_e = end - fs

            g_s = int(fs/2)
            g_e = int(end - fs/2)

            b_s = 0
            b_e = end
        else:
            r_s = 0
            r_e = end

            g_s = int(fs / 2)
            g_e = int(end - fs / 2)

            b_s = fs
            b_e = end - fs

        return ((b_s,b_e), (g_s,g_e), (r_s,r_e))

    def overlap(self):
        # ((7, 15), (3, 11), (0, 8))
        ####   red
         ####  green
          #### blue
        # ((0, 8), (3, 11), (7, 15))
          #### red
         ####  green
        ####   blue
        buffer_len = len(self.vb.buffer)
        # Amount of frames to remove from full buffer
        frame_split = int((buffer_len / 2) + 1)
        gap = buffer_len - frame_split

        if self.reverse:
            r_index = (0, frame_split)
            g_index = (int(gap / 2), int(buffer_len - (gap / 2)))
            b_index = (gap, buffer_len)

            return (b_index, g_index, r_index)
        else:
            r_index = (gap, buffer_len)
            g_index = (int(gap / 2), int(buffer_len - (gap / 2)))
            b_index = (0, frame_split)

            return (b_index, g_index, r_index)

    def render(self):
        frame_arr = self.vb.getArray()
        idx = self.idx()

        if len(frame_arr[0].shape) == 3:
            r = np.array(self.filt(frame_arr[idx[2][0]:idx[2][1], :, :, 2], axis=0), np.uint8)
            g = np.array(self.filt(frame_arr[idx[1][0]:idx[1][1], :, :, 1], axis=0), np.uint8)
            b = np.array(self.filt(frame_arr[idx[0][0]:idx[0][1], :, :, 0], axis=0), np.uint8)
            bgr = cv2.merge((b,g,r))
            return bgr
        else:
            r = np.array(self.filt(frame_arr[idx[2][0]:idx[2][1], :, :], axis=0), np.uint8)
            g = np.array(self.filt(frame_arr[idx[1][0]:idx[1][1], :, :], axis=0), np.uint8)
            b = np.array(self.filt(frame_arr[idx[0][0]:idx[0][1], :, :], axis=0), np.uint8)
            bgr = cv2.merge((b, g, r))
            return bgr

class SlideScan:
    def __init__(self,count=None,vert=False,reverse=False,pixels=1):
        self.pixel = pixels
        self.vert = vert
        self.reverse = reverse
        self.width = None
        self.height = None
        self.count = count if count and count > 0 else None

        self.bg = None
        self.bg_h = None
        self.bg_w = None
        self.pos = None
        self.done = False

    def generateBlank(self):
        if self.vert:
            self.bg_w = self.width
            self.bg_h = self.height + (self.pixel * self.count)
        else:
            self.bg_w = self.width + (self.pixel * self.count)
            self.bg_h = self.height

        return np.zeros((self.bg_h, self.bg_w, 3), np.uint8)

    def bgInit(self,frame):
        fr_shape = frame.shape
        self.width = fr_shape[1]
        self.height = fr_shape[0]
        big = self.width if self.width >= self.height else self.height
        self.count = self.count if self.count else big
        self.bg = self.generateBlank()
        if self.vert:
            self.pos = self.bg_h if self.reverse else 0
        else:
            self.pos = self.bg_w if self.reverse else 0

    def shifts(self):
        # creates slices for pastes
        # if reverse we subtract shift value, else we add shift value
        # if vert we do range check with height, else with width
        if self.vert:
            if self.reverse:
                w1 = 0
                w2 = self.width
                h1 = self.pos - self.height
                h2 = self.pos
                self.pos -= self.pixel
                if (self.pos-self.heigh) < 0:
                    self.done = True
                return (h1, h2, w1, w2)
            else:
                w1 = 0
                w2 = self.width
                h1 = self.pos
                h2 = self.pos + self.height

                self.pos += self.pixel
                if self.pos+self.height > self.bg_h:
                    self.done = True
                return (h1,h2,w1,w2)
        else:
            if self.reverse:
                w1 = self.pos - self.width
                w2 = self.pos
                h1 = 0
                h2 = self.height
                self.pos -= self.pixel
                if (self.pos - self.width) < 0:
                    self.done = True
                return (h1, h2, w1, w2)
            else:
                w1 = self.pos
                w2 = self.pos + self.width
                h1 = 0
                h2 = self.height
                self.pos += self.pixel
                if self.pos+self.width > self.bg_w:
                    self.done = True
                return (h1, h2, w1, w2)


    def update(self, frame):
        # first update we set the width/height and generate a bg canvas
        if type(self.bg) == type(None):
            self.bgInit(frame)


            s = self.shifts()
            self.bg[s[0]:s[1], s[2]:s[3], :] = frame
        else:
            s = self.shifts()
            self.bg[s[0]:s[1],s[2]:s[3],:] = frame

    def render(self):
        return self.bg

