from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
def download(url):
    yt = YouTube(url, on_progress_callback = on_progress)
    # print(yt.title)
    ys = yt.streams.get_highest_resolution()
    x = ys.download('video')
    print(x)
    os.rename(x,'video/video.mp4')
