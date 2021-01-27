import cv2
import numpy as np
import os

def vidIN(fp):
    # TODO:
    pass

def vidOUT():
    # TODO:
    pass

def imgIN():
    # TODO:
    pass

def imgOUT():
    # TODO:
    pass

class VideoBuffer:
    def __init__(self,max_size=15):
        self.buffer = []
        self.buffer_max = max_size

    def update(self, img_input):
        self.buffer.append(img_input)
        if len(self.buffer) > self.buffer_max:
            del self.buffer[0]

    def getBuffer(self):
        return self.buffer

class VideoBufferSingle:
    def __init__(self):
        self.last = None

    def update(self, img_input):
        self.last = img_input

    def getLast(self):
        return self.last

def translate(frame,resize=1.0,rotate=0,crop_dim=None):
    if resize != 1.0:
        frame = cv2.resize(frame,(0,0),fx=resize,fy=resize)

    if rotate != 0:
        for i in range(rotate):
            frame = np.rot90(frame)

    if type(crop_dim) != type(None):
        x = crop_dim
        frame = frame[x[0]:x[1],x[2]:x[3],:]

    return frame

def fnCreateBasicVid(vid_fp):
    # TODO: create extended filepath with something to keep from constantly overwriting
    pass

def videoMetadata(vid_fp):
    cap = cv2.VideoCapture(vid_fp)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return {
        'width':width,
        'height':height,
        'fps':fps,
        'count':count,
        'fourcc':fourcc
    }

def isImage(fp):
    img_ext = ['jpg','jpeg','png']
    rt, ext = os.path.splitext(fp)
    if ext.lstrip('.').lower() in img_ext:
        return True
    else:
        return False

def isVideo(fp):
    vid_ext = ['mp4','mov','avi']
    rt, ext = os.path.splitext(fp)
    if ext.lstrip('.').lower() in vid_ext:
        return True
    else:
        return False

class imgFolderToVideo:
    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.frame_fps = os.listdir(folder_path)
        self.current = None
        self.frame_pos = 0
        self.done = False

    def __next__(self):
        self.read_frame()
        return self.done, self.current

    def read_frame(self):
        frame = cv2.imread(os.path.join(self.folder_path,self.frame_fps[self.frame_pos]))
        self.frame_pos += 1
        if self.frame_pos >= len(self.frame_fps):
            self.done = True

        self.current = frame

