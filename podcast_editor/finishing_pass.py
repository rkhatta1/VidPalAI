import os
import sys
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- CONFIGURATION ---
DIRECTOR_EDITS_PATH = 'output/director_edits.json'
FINAL_XML_PATH = 'output/final_cut_multicam.fcpxml'
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
    
    fmt_id = "r1"
    ET.SubElement(resources, 'format', id=fmt_id, name=f"FFVideoFormat1080p{FRAME_RATE}", frameDuration=f"100/{FRAME_RATE * 100}s", width="1920", height="1080")

    asset_ids = {}
    for i, (cam_id, path) in enumerate(video_sources.items()):
        asset_id = f"r{i+2}"
        asset_ids[cam_id] = asset_id
        abs_path = os.path.abspath(path)
        ET.SubElement(resources, 'asset', id=asset_id, name=cam_id, src=f"file://{abs_path}", hasVideo="1", format=fmt_id)

    # --- 2. Create the Multicam Clip Resource ---
    # [FIX] Manually create a unique ID for the multicam resource.
    multicam_id = f"r{len(asset_ids) + 2}" 
    multicam = ET.SubElement(resources, 'multicam', id=multicam_id, name="AI_Multicam_Clip", format=fmt_id)
    
    for cam_id, asset_id in asset_ids.items():
        mc_angle = ET.SubElement(multicam, 'mc-angle', name=cam_id, angleID=cam_id)
        ET.SubElement(mc_angle, 'video', ref=asset_id, lane="1", offset="0s", duration="21600000/600s")

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
    
    # [FIX] Use the manually created ID variable to set the reference.
    mc_clip.set('ref', multicam_id)

    for cut in full_edl:
        start_time = cut['start_time']
        end_time = cut['end_time']
        camera_id = cut['camera_id']
        
        duration_rational = f"{int((end_time - start_time) * FRAME_RATE)}/{FRAME_RATE}s"
        
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

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load director's edits
    director_edits_full_path = os.path.join(project_root, DIRECTOR_EDITS_PATH)
    print(f"Loading director edits from {director_edits_full_path}...")
    try:
        with open(director_edits_full_path, 'r') as f:
            director_edits_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Director edits file not found. Please run director_pass.py first.")
        return

    print("Stitching all chapter edits into a single timeline...")
    full_edl = []
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

    final_xml_full_path = os.path.join(project_root, FINAL_XML_PATH)
    with open(final_xml_full_path, "w") as f:
        f.write(fcpxml_content)
        
    print(f"\n🎉 Success! Multi-Camera EDL saved to: {final_xml_full_path}")
    print("You can now import this file directly into Premiere Pro.")
    print("--- Finishing Pass Finished ---")


if __name__ == '__main__':
    # Add project root to path to handle running from different directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    run_finishing_pass()
