import cv2
import numpy as np
import argparse, os

parser = argparse.ArgumentParser(description='Statistical function across a number of video frames')
parser.add_argument('-i','--input',required=True,help='Filepath for video input')
parser.add_argument('-b','--buffer_size',help='Amount of frames to condense',default=5,type=int)
parser.add_argument('-f','--filter',help='Filter type',choices=('mean','median','min','max'),default='median')
args = vars(parser.parse_args())

cap = cv2.VideoCapture(args['input'])
fn_pre = args['input'].split('.')[:-1]
fn = os.path.abspath(f"{'.'.join(fn_pre)}_{args['filter']}.mp4")
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter(fn,fourcc,fps,(w,h))
buffer = []

print(f"output:{fn} {w}x{h}@{fps}")

while True:
    ret, frame = cap.read()
    if ret:
        buffer.append(frame)
        if len(buffer) > args['buffer_size']:
            del buffer[0]

        if args['filter'].lower() == 'median':
            out_frame = np.array(np.median(np.array(buffer),0),np.uint8)
        elif args['filter'].lower() == 'mean':
            out_frame = np.array(np.mean(np.array(buffer),0),np.uint8)
        elif args['filter'].lower() == 'min':
            out_frame = np.array(np.min(np.array(buffer),0),np.uint8)
        elif args['filter'].lower() == 'max':
            out_frame = np.array(np.max(np.array(buffer),0),np.uint8)
        else:
            print('Filter type not recognized')
            break

        out.write(out_frame)
    else:
        break

out.release()
cap.release()
