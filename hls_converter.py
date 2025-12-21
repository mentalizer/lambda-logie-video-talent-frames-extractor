#!/usr/bin/env python3
"""
Simple HLS to MP4 converter using ffmpeg
"""
import ffmpeg
import tempfile
import os

def convert_hls_to_mp4(hls_url: str, output_path: str = None) -> str:
    """
    Convert HLS stream to MP4 using ffmpeg

    Args:
        hls_url: HLS .m3u8 playlist URL
        output_path: Optional output path, defaults to temp file

    Returns:
        Path to converted MP4 file
    """
    if not output_path:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")

    print(f"🎬 Converting HLS to MP4: {hls_url}")

    try:
        # Use ffmpeg to convert HLS to MP4
        stream = ffmpeg.input(hls_url)
        stream = ffmpeg.output(stream, output_path,
                              vcodec='copy',  # Copy video codec (no re-encoding)
                              acodec='copy',  # Copy audio codec (no re-encoding)
                              avoid_negative_ts='make_zero')

        # Run ffmpeg
        ffmpeg.run(stream, overwrite_output=True, quiet=False)

        # Check output
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            size = os.path.getsize(output_path)
            print(f"✅ Conversion successful: {size} bytes")
            return output_path
        else:
            raise Exception("Output file not created or empty")

    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if e.stderr else "Unknown error"
        print(f"❌ FFmpeg error: {stderr}")
        raise Exception(f"FFmpeg conversion failed: {stderr}")

if __name__ == "__main__":
    # Test with a sample HLS URL
    test_url = "https://example.com/playlist.m3u8"
    print(f"Testing HLS conversion with: {test_url}")
    print("Note: This is just a template - replace with real HLS URL")

    # Uncomment to test:
    # output = convert_hls_to_mp4(test_url)
    # print(f"Output: {output}")
