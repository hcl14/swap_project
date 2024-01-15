import sys
import cv2
#from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import traceback
import decord
import time
import ffmpeg


def get_video_data(video_file):

    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return length, height, width, fps

# According to this, decord_batch_cpu is one of the top benchmarks
# https://github.com/bml1g12/benchmarking_video_reading_python

# alternative is to use nvidia dali and stream immediately into tensor
# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/sequence_processing/video/video_reader_label_example.html

# This is simple call of decord, VideoReader instead of VideoLoader
# decord produces RGB frames
class VideoReader:
    def __init__(self, video_file, get_all_at_once=True, resize=None, unbatch=False, batch_size=10, device='cpu'):

        self.video_file = video_file
        self.unbatch = unbatch
        self.batch_size = batch_size

        # make exception parsing, so we can contiue if video hasn't been loaded
        try:

            start_fr = time.time()
            self.n_frames, self.height, self.width, self.fps = get_video_data(self.video_file)
            print("Getting metadata took", time.time() - start_fr)

            # decord supports instant resize
            if resize is not None:
                self.height, self.width = resize

            self.np_arr_shape = (self.height, self.width, 3)

            if device == "gpu":
                ctx = decord.gpu(0)
            else:
                ctx = decord.cpu()

            self.video_loader = decord.VideoReader(self.video_file, width=self.width, height=self.height, ctx=ctx)

            assert len(self.video_loader) == self.n_frames

            self.error = False
        except:
            tb = traceback.format_exc()
            print("Exception in video conversion process")
            print(tb)
            self.error = True

        # calculate number of batches
        self.total, remainder = divmod(self.n_frames,batch_size)
        if remainder:
            self.total += 1

        if get_all_at_once:
            self.get_all()

        # getting audio
        #start = time.time()
        #self.audio = AudioFileClip(self.video_file)
        #print("Extracting audio took", time.time()-start)  # 0.3 sec

    def get_all(self): # be careful when on gpu?
        start = time.time()
        self.frames = self.video_loader[:].asnumpy()
        if self.unbatch:
            start_unb = time.time()
            self.frames = list(self.frames)
            print("Unbatch took", time.time() - start_unb)
        assert len(self.frames) > 0
        print(f"Read {self.n_frames} frames with shape {self.np_arr_shape} in {time.time()-start}s")

    def get_batch(batch_idx): # get specific batch
        data = video_loader.get_batch(list(range(batch_idx,min(batch_idx+batch_size, self.n_frames)))).asnumpy() # asnumpy() greatly increases time overhead
        return data

# optimized ffmpeg wrapper video saving with source video as audio source
# Modified from https://github.com/vujadeyoon/Fast-Video-Processing/blob/master/vujade/vujade_videocv.py
class VideoWriterFFmpeg:
    def __init__(self, path_video, audio_source=None, resolution=(1080, 1920), fps=30.0, _qp_val=0, pix_fmt='bgr24', _codec='libx264'):
        if path_video is None:
            raise ValueError('path_video should be assigned')

        self.path_video = path_video
        self.height = int(resolution[0])
        self.width = int(resolution[1])
        self.fps = float(fps)
        self.qp_val = _qp_val

        self.pix_fmt = pix_fmt # input pixel format, bgr is faster
        #self.pix_fmt = 'rgb24' # 1.4 sec -> 2.1 sec !!!
        self.codec = _codec

        params = {'c:v': self.codec}
        # https://stackoverflow.com/questions/71018371/ffmpeg-python-audio-getting-dropped-in-final-video

        if audio_source is not None:
            self.audio_stream = ffmpeg.input(audio_source).audio  # Source audio stream
            params['c:a'] ='copy'
        else:
            self.audio_stream = None # DEBUG


        self.wri = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt=self.pix_fmt, s='{}x{}'.format(self.width, self.height))
            .filter('fps', fps=self.fps, round='up')
            .output(self.audio_stream, self.path_video, pix_fmt='yuv420p', **params, **{'qscale:v': self.qp_val})
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def imwrite(self, _list_img):
        for idx, img in enumerate(_list_img):
            self.wri.stdin.write(img)

    def close(self):
        self.wri.stdin.close()
        self.wri.wait()
'''
# simple video saving, which takes 6 sec even without audio
def simple_video_save(out_filename, frames, fps):
    # RGB order
    frames = [x[..., ::-1] for x in frames]
    video_clips = ImageSequenceClip(frames, fps=fps)
    video_clips.write_videofile(out_filename, remove_temp=True)
'''


if __name__ == '__main__':

    fname = sys.argv[1]

    v = VideoReader(fname)
    # check frame
    cv2.imwrite('frame0.png', v.frames[0][...,::-1])

    start = time.time()

    writer = VideoWriterFFmpeg('out.mp4', audio_source=fname, resolution=(v.height, v.width), fps=v.fps, pix_fmt='rgb24') # 2 sec
    writer.imwrite(v.frames)
    writer.close()

    #simple_video_save('out.mp4', v.frames, v.fps) # Writing video took 6.341479301452637

    print("Writing video took", time.time() - start)


