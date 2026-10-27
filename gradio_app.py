import os
from pathlib import Path
import gradio as gr
import subprocess
import json
import cv2
import tempfile
from PIL import Image
import numpy as np

CONFIG = {
    "model_path": "OpenGVLab/InternVL2-8B",
    "get_boundary_model_path": "Yongxin-Guo/trace-uni",
    "get_mask_model_path": "./checkpoints/sam2.1_hiera_base_plus.pt",
    "output_folder": "./results/",
    "frame_count": 16,
}

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cap.release()
    height, width = image.shape[:2]
    if height > 750:
        scale = 750 / height
        new_width = int(width * scale)
        new_height = 750
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def run_inference_pipeline(video_path, bbox):
    """
    Run the entire inference pipeline for video processing
    """
    # Ensure output folder exists
    os.makedirs(CONFIG["output_folder"], exist_ok=True)

    # Prepare file paths
    video_name = os.path.basename(video_path)
    qa_file_path = os.path.join(
        CONFIG["output_folder"], f"{os.path.splitext(video_name)[0]}_boundary.json"
    )
    final_json_path = os.path.join(
        CONFIG["output_folder"],
        f"{os.path.splitext(video_name)[0]}_boundary_caption.json",
    )
    final_video_path = os.path.join(
        CONFIG["output_folder"],
        f"{os.path.splitext(video_name)[0]}_boundary_caption.mp4",
    )
    masked_video_path = os.path.join(
        CONFIG["output_folder"], f"{os.path.splitext(video_name)[0]}_mask.mp4"
    )
    print(f"Final JSON Path: {final_json_path}")
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    h,w = frame.shape[:2]
    print(h,w)
    video.release()
    bbox = [int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)]
    print(bbox)
    object_bbox_path = Path(CONFIG['output_folder'])/f"{os.path.splitext(video_name)[0]}_bbox.txt"
    with open(object_bbox_path, "w") as f:
        f.write(','.join(map(str, bbox)))
    commands = [
        # Step 1: Parsing/Boundary Detection
        f"python -m scripts.get_boundary "
        f"--video_paths {video_path} "
        f"--questions 'Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences.' "
        f"--model_path {CONFIG['get_boundary_model_path']}",
    ]


    commands.append(
        f"python scripts/get_masks.py "
        f"--video_path {video_path} "
        f"--txt_path {object_bbox_path} "
        f"--model_path {CONFIG['get_mask_model_path']} "
        f"--video_output_path {CONFIG['output_folder']} "
        f"--save_to_video True"
    )

    # Step 2: Captioning
    commands.append(
        f"python scripts/get_caption.py "
        f"--model_path {CONFIG['model_path']} "
        f"--QA_file_path {qa_file_path} "
        f"--video_folder {CONFIG['output_folder']} "
        f"--answers_output_folder {CONFIG['output_folder']} "
        f"--extract_frames_method max_frames_num "
        f"--max_frames_num {CONFIG['frame_count']} "
        f"--frames_from video "
        f"--final_json_path {final_json_path} "
        f"--provide_boundaries"
    )

    # Step 3: Generate Visualization
    commands.append(
        f"python scripts/get_vis.py {masked_video_path if object_bbox_path else video_path} {final_json_path} {final_video_path}"
    )

    # Execute commands
    for cmd in commands:
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in command: {cmd}")
            print(f"Error details: {e}")
            return None

    try:
        with open(final_json_path, "r") as f:
            results = json.load(f)
        return {"captions": results, "final_video": final_video_path}
    except Exception as e:
        print(f"Error reading results: {e}")
        return None

def get_bounding_box(image):
    alpha_channel = image[:, :, 3]
    y_coords, x_coords = np.where(alpha_channel > 0)
    
    if y_coords.size == 0 or x_coords.size == 0:
        return None 
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min_ratio = x_min / image.shape[1]
    x_max_ratio = x_max / image.shape[1]
    y_min_ratio = y_min / image.shape[0]
    y_max_ratio = y_max / image.shape[0]
    return x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio
def caption_video(video, edited_image):
    """
    Gradio-friendly wrapper for inference pipeline
    video: path to the uploaded video
    bbox_file: path to the uploaded bounding box file (optional)
    edited_image: the edited first frame image returned by ImageEditor (PIL Image)
    """
    layer_0 = edited_image['layers'][0]
    bbox = get_bounding_box(layer_0)

    if video is None:
        return "Please upload a video first.", None
    results = run_inference_pipeline(video, bbox)

    if results is None:
        return "Processing failed. Please check the logs.", None

    # Format captions nicely
    captions_text = "\n\n".join(
        [
            f"Event {i+1} (Time: {event.get('timestamp', 'N/A')}):\n{event.get('model_answer', 'No caption')}"
            for i, event in enumerate(results.get("captions", []))
        ]
    )

    return captions_text, results.get("final_video")




def create_demo():
    """
    Create Gradio interface
    """

    DESCRIPTION = """# CAT2: 
    This is a demo for our 'CAT2' [paper](https://github.com/yoloytang/CAT-V).
    Code is available [here](https://github.com/yoloytang/CAT-V).
    This demo performs captioning with optional object bounding box annotation.
    """

    with gr.Blocks() as demo:
        gr.Markdown("# Caption Anything Demo")
        gr.Markdown(DESCRIPTION)
        gr.Markdown(
            "Upload a video and optionally a bounding box file. Or draw a rectangle on the first frame of the video to provide a bounding box. (Note: The ImageEditor does not return bounding box coordinates directly. Further processing may be required.)"
        )

        with gr.Row():
            video_input = gr.Video(label="Upload Video",height=800)
            first_frame_editor = gr.ImageEditor(label="Draw a rectangle on the First Frame",height=800)

        video_input.change(fn=extract_first_frame, inputs=video_input, outputs=first_frame_editor)

        caption_button = gr.Button("Generate Captions")

        output_text = gr.Textbox(label="Video Captions")
        output_video = gr.Video(label="Processed Video")

        caption_button.click(
            fn=caption_video,
            inputs=[video_input, first_frame_editor],
            outputs=[output_text, output_video],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # Make accessible from other machines
        server_port=8889,
        debug=True,
    )
