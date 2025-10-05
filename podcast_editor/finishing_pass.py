import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import sys

# --- CONFIGURATION ---
DIRECTOR_EDITS_PATH = 'output/director_edits.json'
FINAL_XML_PATH = 'output/final_cut_multicam.fcpxml'
SOURCE_VIDEOS = {
    "cam_host": "input/cam_host.mp4",
    "cam_guest": "input/cam_guest.mp4",
    "cam_wide": "input/cam_wide.mp4"
}
FRAME_RATE = 30
# Define the absolute root path of your project on the WINDOWS machine.
# Use forward slashes.
WINDOWS_PROJECT_ROOT = "E:/Random/VidPal"

def generate_flattened_fcpxml(full_edl, video_sources):
    """
    Generates a flattened FCPXML with individual clips (Resolve-compatible).
    """
    print("Generating Flattened FCPXML structure for Resolve...")
    
    fcpxml = ET.Element('fcpxml', version='1.10')

    # --- Resources Section ---
    resources = ET.SubElement(fcpxml, 'resources')
    
    fmt_id = "r1"
    ET.SubElement(
        resources, 'format',
        id=fmt_id,
        name=f"FFVideoFormat1080p{FRAME_RATE}",
        frameDuration=f"100/{FRAME_RATE * 100}s",
        width="1920",
        height="1080"
    )

    asset_ids = {}
    
    # Calculate total duration for asset metadata
    total_duration_seconds = full_edl[-1]['end_time'] if full_edl else 0
    total_duration_rational = f"{int(total_duration_seconds * FRAME_RATE)}/{FRAME_RATE}s"
    
    for i, (cam_id, relative_path) in enumerate(video_sources.items()):
        asset_id = f"r{i+2}"
        asset_ids[cam_id] = asset_id
        
        # Construct Windows File URI
        windows_path = f"{WINDOWS_PROJECT_ROOT}/{relative_path.replace('//', '/')}"
        video_uri = f"file:///{windows_path}"
        
        ET.SubElement(
            resources, 'asset',
            id=asset_id,
            name=cam_id,
            hasVideo="1",
            hasAudio="1",
            format=fmt_id,
            duration=total_duration_rational
        )
        # Add media-rep child
        ET.SubElement(
            resources.find(f"./asset[@id='{asset_id}']"),
            'media-rep',
            kind="original-media",
            src=video_uri
        )

    # --- Library/Event/Project/Sequence Structure ---
    library = ET.SubElement(fcpxml, 'library')
    event = ET.SubElement(library, 'event', name="AI Podcast Edit")
    
    project = ET.SubElement(event, 'project', name="AI Edited Podcast (Flattened)")
    sequence = ET.SubElement(
        project, 'sequence',
        format=fmt_id,
        duration=total_duration_rational,
        tcStart="0s",
        tcFormat="NDF"
    )
    spine = ET.SubElement(sequence, 'spine')

    # --- Generate Individual Clips ---
    for cut in full_edl:
        start_time = cut['start_time']
        end_time = cut['end_time']
        camera_id = cut['camera_id']
        
        # Calculate durations and offsets in rational time
        offset_rational = f"{int(start_time * FRAME_RATE)}/{FRAME_RATE}s"
        duration_rational = f"{int((end_time - start_time) * FRAME_RATE)}/{FRAME_RATE}s"
        
        # Create clip element
        clip = ET.SubElement(
            spine, 'clip',
            name=camera_id,
            offset=offset_rational,
            duration=duration_rational,
            format=fmt_id
        )
        
        # Add video element
        ET.SubElement(
            clip, 'video',
            ref=asset_ids[camera_id],
            start=offset_rational,
            duration=duration_rational
        )
        
        # Add audio element
        ET.SubElement(
            clip, 'audio',
            ref=asset_ids[camera_id],
            start=offset_rational,
            duration=duration_rational
        )

    rough_string = ET.tostring(fcpxml, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def run_finishing_pass():
    print("--- Starting Finishing Pass (Pass 3) ---")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    full_edl.sort(key=lambda x: x['start_time'])

    if not full_edl:
        print("Error: No valid cuts were found to process. Aborting.")
        return
        
    print(f"Successfully compiled {len(full_edl)} cuts into one timeline.")
    fcpxml_content = generate_flattened_fcpxml(full_edl, SOURCE_VIDEOS)
    final_xml_full_path = os.path.join(project_root, FINAL_XML_PATH)
    with open(final_xml_full_path, "w") as f:
        f.write(fcpxml_content)
        
    print(f"\n🎉 Success! Flattened FCPXML saved to: {final_xml_full_path}")
    print("--- Finishing Pass Finished ---")


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    run_finishing_pass()
