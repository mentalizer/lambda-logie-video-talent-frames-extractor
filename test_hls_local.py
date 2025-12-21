#!/usr/bin/env python3
"""
Local test script to verify HLS downloading and MP4 conversion works
"""
import requests
import m3u8
import tempfile
import os
import urllib.parse

def download_hls_to_mp4(hls_url: str, output_path: str) -> bool:
    """
    Download HLS stream and convert to MP4 for testing
    """
    try:
        print(f"Testing HLS URL: {hls_url}")

        # Download and parse HLS playlist
        print("Downloading HLS playlist...")
        response = requests.get(hls_url, timeout=10)
        response.raise_for_status()
        print(f"Playlist size: {len(response.text)} characters")

        # Parse the playlist
        playlist = m3u8.loads(response.text)
        print(f"Found {len(playlist.segments)} segments")

        if not playlist.segments:
            print("No segments found - checking for variant playlist...")
            if playlist.playlists:
                print(f"Found {len(playlist.playlists)} variants")
                for i, variant in enumerate(playlist.playlists[:3]):
                    print(f"  Variant {i}: {variant.stream_info.bandwidth if variant.stream_info else 'unknown'} bandwidth")
            return False

        # Test downloading first segment
        if playlist.segments:
            first_segment = playlist.segments[0]
            base_url = urllib.parse.urljoin(hls_url, '.')
            segment_url = urllib.parse.urljoin(base_url, first_segment.uri)

            print(f"Testing first segment: {segment_url}")
            segment_response = requests.get(segment_url, timeout=10)
            segment_response.raise_for_status()

            print(f"✅ First segment downloaded: {len(segment_response.content)} bytes")
            print(f"Content type: {segment_response.headers.get('content-type', 'unknown')}")

            # Check if it's actually a TS file
            if segment_response.content.startswith(b'G'):  # TS files start with 'G'
                print("✅ Segment appears to be valid MPEG-TS format")
            else:
                print("⚠️ Segment might not be MPEG-TS format")

            return True

    except Exception as e:
        print(f"❌ HLS test failed: {e}")
        return False

def test_amazon_hls_url():
    """Test with a real Amazon HLS URL"""
    # Use the URL from the user's data
    hls_url = "https://m.media-amazon.com/images/S/vse-vms-transcoding-artifact-us-east-1-prod/f07b855f-e4aa-4e28-9b92-d305b5b91ce2/default.jobtemplate.hls.m3u8"

    print("=== Testing Amazon HLS URL ===")
    success = download_hls_to_mp4(hls_url, "test_output.mp4")

    if success:
        print("✅ HLS URL is accessible and segments are downloadable")
    else:
        print("❌ HLS URL test failed")

    return success

def test_mp4_from_hls_pattern():
    """Test if MP4 URLs exist using HLS URL pattern"""
    hls_url = "https://m.media-amazon.com/images/S/vse-vms-transcoding-artifact-us-east-1-prod/f07b855f-e4aa-4e28-9b92-d305b5b91ce2/default.jobtemplate.hls.m3u8"
    mp4_url = hls_url.replace('.m3u8', '.mp4')

    print(f"\n=== Testing MP4 URL from HLS pattern ===")
    print(f"MP4 URL: {mp4_url}")

    try:
        response = requests.head(mp4_url, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ MP4 URL exists and is accessible!")
            return True
        else:
            print(f"❌ MP4 URL returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MP4 URL test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing HLS processing locally...\n")

    # Test MP4 pattern first (easier)
    mp4_exists = test_mp4_from_hls_pattern()

    if not mp4_exists:
        # If MP4 doesn't exist, test HLS processing
        hls_works = test_amazon_hls_url()
        if hls_works:
            print("\n✅ HLS processing is technically possible")
            print("⚠️ But concatenating TS segments may not create valid MP4 files")
            print("💡 Consider using ffmpeg or finding actual MP4 download URLs")
        else:
            print("\n❌ Both MP4 and HLS approaches failed")
            print("🔍 Need to find the correct Amazon Live video download URLs")
    else:
        print("\n🎯 MP4 pattern works! Use this approach instead of HLS processing.")
