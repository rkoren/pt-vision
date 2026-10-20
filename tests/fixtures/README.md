# Test fixtures

`clip_2s_480p.mp4` — seconds 1–3 of `Sports2D/Demo/demo.mp4` from
https://github.com/davidpagnon/Sports2D (BSD 3-Clause, David Pagnon), rescaled to 480p
with `ffmpeg -ss 1 -t 2 -vf scale=-2:480 -c:v libx264 -crf 28`. A person exercising, filmed
side-on. Used by the `model` integration test and the video-probe tests.

Everything else the tests need is generated (`tests/synthetic.py`). Never commit patient video.
