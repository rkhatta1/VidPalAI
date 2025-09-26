import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- CONFIGURATION ---
DIRECTOR_EDITS_PATH = 'output/director_edits.json'
FINAL_XML_PATH = 'output/final_cut.fcpxml'
SOURCE_VIDEO_PATH = 'input/podcast_video_h264_gpu.mp4' # Relative path to the video
VIDEO_FORMAT_ID = "r1"
VIDEO_ASSET_ID = "r2"
FRAME_RATE = 30 # Frames per second

def generate_fcpxml(full_edl, video_path):
    """
    Generates an FCPXML string from a compiled Edit Decision List.

    Args:
        full_edl (list): A list of all cut decisions, sorted by start time.
        video_path (str): The relative path to the source video file.

    Returns:
        str: A pretty-printed FCPXML string.
    """
    print("Generating FCPXML structure...")
    
    # Get absolute path for the video file for the XML src attribute
    abs_video_path = os.path.abspath(video_path)
    
    # FCPXML structure starts here
    fcpxml = ET.Element('fcpxml', version='1.9')

    # --- Resources Section ---
    resources = ET.SubElement(fcpxml, 'resources')
    fmt = ET.SubElement(resources, 'format', id=VIDEO_FORMAT_ID, name=f"FFVideoFormat1080p{FRAME_RATE}", frameDuration=f"100/{FRAME_RATE * 100}s", width="1920", height="1080")
    asset = ET.SubElement(resources, 'asset', id=VIDEO_ASSET_ID, name="Source Video", src=f"file://{abs_video_path}", hasVideo="1", format=VIDEO_FORMAT_ID)

    # --- Library and Project Setup ---
    library = ET.SubElement(fcpxml, 'library')
    event = ET.SubElement(library, 'event', name="AI Podcast Edit")
    
    total_duration_seconds = full_edl[-1]['end_time'] if full_edl else 0
    total_duration_rational = f"{int(total_duration_seconds * FRAME_RATE)}/{FRAME_RATE}s"
    
    project = ET.SubElement(event, 'project', name="AI Edited Podcast")
    sequence = ET.SubElement(project, 'sequence', format=VIDEO_FORMAT_ID, duration=total_duration_rational, tcFormat="NDF")
    spine = ET.SubElement(sequence, 'spine')

    # --- Spine Section (Timeline) ---
    for cut in full_edl:
        start_time = cut['start_time']
        end_time = cut['end_time']
        camera_id = cut['camera_id']
        
        # Calculate durations and offsets in rational format for FCPXML
        offset_rational = f"{int(start_time * FRAME_RATE)}/{FRAME_RATE}s"
        duration_rational = f"{int((end_time - start_time) * FRAME_RATE)}/{FRAME_RATE}s"
        start_rational = f"{int(start_time * FRAME_RATE)}/{FRAME_RATE}s"

        # Each cut is an asset-clip in the spine
        asset_clip = ET.SubElement(spine, 'asset-clip', 
                                   name=camera_id, 
                                   ref=VIDEO_ASSET_ID,
                                   offset=offset_rational,
                                   duration=duration_rational,
                                   start=start_rational,
                                   format=VIDEO_FORMAT_ID)
                                   
    # Convert the ElementTree object to a pretty-printed string
    rough_string = ET.tostring(fcpxml, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def run_finishing_pass():
    """
    Stitches together director edits and converts them into a final
    FCPXML file for video editing software.
    """
    print("--- Starting Finishing Pass (Pass 3) ---")

    # --- 1. Load the Director's Edits ---
    print(f"Loading director edits from {DIRECTOR_EDITS_PATH}...")
    try:
        with open(DIRECTOR_EDITS_PATH, 'r') as f:
            director_edits_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Director edits file not found. Please run director_pass.py first.")
        return

    # --- 2. Stitch all cuts into a single timeline ---
    print("Stitching all chapter edits into a single timeline...")
    full_edl = []
    for chapter_edit in director_edits_data:
        # Assumes the JSON from Gemini has a 'cuts' key as per the prompt
        if 'edl' in chapter_edit and 'cuts' in chapter_edit['edl']:
            full_edl.extend(chapter_edit['edl']['cuts'])
        else:
            print(f"Warning: Chapter '{chapter_edit.get('chapter_title')}' is missing a valid 'edl' with 'cuts'. Skipping.")
    
    # Sort by start time to ensure chronological order
    full_edl.sort(key=lambda x: x['start_time'])

    if not full_edl:
        print("Error: No valid cuts were found to process. Aborting.")
        return
        
    print(f"Successfully compiled {len(full_edl)} cuts into one timeline.")

    # --- 3. Generate and Save FCPXML file ---
    fcpxml_content = generate_fcpxml(full_edl, SOURCE_VIDEO_PATH)

    with open(FINAL_XML_PATH, "w") as f:
        f.write(fcpxml_content)
        
    print(f"\n🎉 Success! Final EDL saved to: {FINAL_XML_PATH}")
    print("You can now import this file into Adobe Premiere Pro or Final Cut Pro.")
    print("--- Finishing Pass Finished ---")

if __name__ == '__main__':
    run_finishing_pass()
