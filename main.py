import cv2
import visual_filters as vf

'''
boiler plate code for reading/watching a video with CV2
'''
cap = cv2.VideoCapture(0)

# Uncomment a single line to test each filter
# t = vf.MotionDetection(shadows=False)
t = vf.HarrisSimple(reverse=False)
# t = vf.HarrisTemporal(buffer_size=15,filter_type='max',align='overlap',reverse=False)
# t = vf.TemporalBuildUp(filter_type='max')
# t = vf.TemporalFilter(buffer_size=15,filter_type='max')
# t = vf.BeltScan(pixels=20,vert=False)
# t = vf.SlitScan(pixels=1,vert=False)
# t = vf.SlideScan(count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), vert=False, reverse=False, pixels=1)

while True:
    ret, frame = cap.read()
    if ret:
        # run filters here
        # frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        t.update(frame)
        filt_frame = t.render()

        cv2.imshow('window',filt_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break