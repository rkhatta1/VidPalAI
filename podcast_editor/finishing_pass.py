import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- CONFIGURATION ---
DIRECTOR_EDITS_PATH = 'output/director_edits.json'
FINAL_XML_PATH = 'output/final_cut_multicam.fcpxml'
# [NEW] Define the source files for the XML
SOURCE_VIDEOS = {
    "cam_host": "input/cam_host.mp4",
    "cam_guest": "input/cam_guest.mp4",
    "cam_wide": "input/cam_wide.mp4"
}
FRAME_RATE = 30 # Frames per second

def generate_multicam_fcpxml(full_edl, video_sources):
    """
    Generates a true multi-camera FCPXML file.
    """
    print("Generating Multi-Camera FCPXML structure...")
    
    fcpxml = ET.Element('fcpxml', version='1.9')

    # --- 1. Resources Section ---
    resources = ET.SubElement(fcpxml, 'resources')
    
    # Define a single format for all clips
    fmt_id = "r1"
    ET.SubElement(resources, 'format', id=fmt_id, name=f"FFVideoFormat1080p{FRAME_RATE}", frameDuration=f"100/{FRAME_RATE * 100}s", width="1920", height="1080")

    # Define each video file as a separate asset
    asset_ids = {}
    for i, (cam_id, path) in enumerate(video_sources.items()):
        asset_id = f"r{i+2}"
        asset_ids[cam_id] = asset_id
        abs_path = os.path.abspath(path)
        ET.SubElement(resources, 'asset', id=asset_id, name=cam_id, src=f"file://{abs_path}", hasVideo="1", format=fmt_id)

    # --- 2. Create the Multicam Clip Resource ---
    multicam = ET.SubElement(resources, 'multicam', name="AI_Multicam_Clip", format=fmt_id)
    
    for cam_id, asset_id in asset_ids.items():
        mc_angle = ET.SubElement(multicam, 'mc-angle', name=cam_id, angleID=cam_id)
        # This clip represents the entire source video for that angle
        ET.SubElement(mc_angle, 'video', ref=asset_id, lane="1", offset="0s", duration="21600000/600s") # Assuming a long duration

    # --- 3. Library and Project Setup ---
    library = ET.SubElement(fcpxml, 'library')
    event = ET.SubElement(library, 'event', name="AI Podcast Edit")
    
    total_duration_seconds = full_edl[-1]['end_time'] if full_edl else 0
    total_duration_rational = f"{int(total_duration_seconds * FRAME_RATE)}/{FRAME_RATE}s"
    
    project = ET.SubElement(event, 'project', name="AI Edited Podcast")
    sequence = ET.SubElement(project, 'sequence', format=fmt_id, duration=total_duration_rational, tcFormat="NDF")
    spine = ET.SubElement(sequence, 'spine')

    # --- 4. Create the Timeline using the Multicam Clip ---
    mc_clip = ET.SubElement(spine, 'mc-clip', offset="0s", name="AI Edited Sequence", duration=total_duration_rational)
    
    # Add the reference to the multicam resource we defined
    mc_clip.set('ref', multicam.attrib['id']) # This will be auto-generated, e.g., 'multicam-1'

    for cut in full_edl:
        start_time = cut['start_time']
        end_time = cut['end_time']
        camera_id = cut['camera_id']
        
        duration_rational = f"{int((end_time - start_time) * FRAME_RATE)}/{FRAME_RATE}s"
        
        # This tag tells the mc-clip which angle to use for this duration
        ET.SubElement(mc_clip, 'mc-source', angleID=camera_id, srcEnable="video", duration=duration_rational)

    # --- 5. Finalize and Pretty-Print XML ---
    rough_string = ET.tostring(fcpxml, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def run_finishing_pass():
    """
    Stitches edits and converts them into a final FCPXML file.
    """
    print("--- Starting Finishing Pass (Pass 3) ---")

    # Load director's edits
    print(f"Loading director edits from {DIRECTOR_EDITS_PATH}...")
    try:
        with open(DIRECTOR_EDITS_PATH, 'r') as f:
            director_edits_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Director edits file not found. Please run director_pass.py first.")
        return

    # Stitch all cuts into a single timeline
    print("Stitching all chapter edits into a single timeline...")
    full_edl = []
    # [MODIFIED] The top-level key from the new director_pass.py is 'director_edits'
    for chapter_edit in director_edits_data.get('director_edits', []):
        if 'edl' in chapter_edit and 'cuts' in chapter_edit['edl']:
            full_edl.extend(chapter_edit['edl']['cuts'])
        else:
            print(f"Warning: Chapter '{chapter_edit.get('chapter_title')}' missing valid cuts. Skipping.")
    
    full_edl.sort(key=lambda x: x['start_time'])

    if not full_edl:
        print("Error: No valid cuts were found to process. Aborting.")
        return
        
    print(f"Successfully compiled {len(full_edl)} cuts into one timeline.")

    # Generate and Save FCPXML
    fcpxml_content = generate_multicam_fcpxml(full_edl, SOURCE_VIDEOS)

    with open(FINAL_XML_PATH, "w") as f:
        f.write(fcpxml_content)
        
    print(f"\n🎉 Success! Multi-Camera EDL saved to: {FINAL_XML_PATH}")
    print("You can now import this file directly into Premiere Pro.")
    print("--- Finishing Pass Finished ---")

if __name__ == '__main__':
    run_finishing_pass()
