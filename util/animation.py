import ffmpeg

def make_animation(input_folder, output_file, input_format="png"):
    (
        ffmpeg
        .input('/content/' + input_folder + '/*.' + input_format, pattern_type='glob', framerate=30)
        .output('/content/' + output_file, vcodec="mpeg4")
        .overwrite_output()
        .run()
    )
