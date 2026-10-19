from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from rb3charts.review_server import create_review_server


def test_review_server_serves_chart_and_audio_but_not_other_files(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "review.html").write_text("<h1>Review</h1>", encoding="utf-8")
    (tmp_path / "audio" / "song.wav").write_bytes(b"song")
    (tmp_path / "private.txt").write_text("private", encoding="utf-8")
    server = create_review_server(tmp_path, host="127.0.0.1", port=0)
    thread = Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/review.html") as response:
            assert response.read() == b"<h1>Review</h1>"
        with urlopen(base + "/audio/song.wav") as response:
            assert response.read() == b"song"
        for path in ("/", "/private.txt", "/../private.txt"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_review_server_requires_a_complete_review_folder(tmp_path):
    with pytest.raises(ValueError, match="review.html and audio/song.wav"):
        create_review_server(tmp_path, host="127.0.0.1", port=0)
