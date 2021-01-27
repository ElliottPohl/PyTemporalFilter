# PyTemporalFilter
opencv utility to apply statistical functions across a number of frames

#FILTERS
current filters:
- motion detection masking pasted onto perma background
- pixel sorting with direction to sort. Need to add ability to adjust strength
- slide is basically a panorama filter, but can be used to create abstracts
- slitscan takes in a video and freezes a single pixel at a time until the frame 
is frozen

#utilities
- filenames: create unique filenames so each file can be saved without overwrite
- getinfo: get metadata from video files
- harris_splits: takes in a frame stack, out puts array slice dimensions for various harris filter styles
- image_is: bool funcs that allow for quick checks of various files
- readwriter: takes in a video, outputs a new video file. Functions to get np.arrays
- translation: resizing and rotating
- video_buffer: maintains a video stack for stat functions

## Apps
### CLI
- Run filters automatically

## IO
###loader
- VIDEO:
    - read video frames
    - frame stack buffer (preload option)
    - output VideoWriter object
- IMAGE:
    - read single images
    - read entire folder of images with ability to resize all to fit
- METADATA:
    - video metadata
    - image metadata

###writer
- VIDEO:
    - write frames to VideoWriter obj
    - preview output
    - option to only out every X number of frames
- IMAGE:
    - save single image
    - save multiple images with incremented names
    
###Filenames
- No input filenames
    - datetime
    - \###.jpg++
- increment an already existing number
    - img001 -> img002
- argprint filename
    - include the arguments passed in the filename

##Utilities
###Filters
- Temporal Filters
    - Min/Max/Mean/Median
    - harris
- Circuit Camera
    - Slit Scan
    - Pano Slide
    - Belt Scan
    - Half Belt Scan
- Pixel Sort
    - Vertical
    - Horizontal
###Translation
- Rotate
- Resize
- Crop
- Paste
