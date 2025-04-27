import spaces
import gradio as gr
import os
import torch
import numpy as np
import cv2
import huggingface_hub
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# # Remove all CUDA-specific configurations
# torch.autocast(device_type="cpu", dtype=torch.float32).__enter__()

def preprocess_image(image):
    return image, gr.State([]), gr.State([]), image

def get_point(point_type, tracking_points, trackings_input_label, first_frame_path, evt: gr.SelectData):
    print(f"You selected {evt.value} at {evt.index} from {evt.target}")
    tracking_points.value.append(evt.index)
    print(f"TRACKING POINT: {tracking_points.value}")
    if point_type == "include":
        trackings_input_label.value.append(1)
    elif point_type == "exclude":
        trackings_input_label.value.append(0)
    print(f"TRACKING INPUT LABEL: {trackings_input_label.value}")
    transparent_background = Image.open(first_frame_path).convert('RGBA')
    w, h = transparent_background.size
    fraction = 0.02
    radius = int(fraction * min(w, h))
    transparent_layer = np.zeros((h, w, 4), dtype=np.uint8)
    for index, track in enumerate(tracking_points.value):
        if trackings_input_label.value[index] == 1:
            cv2.circle(transparent_layer, track, radius, (0, 255, 0, 255), -1)
        else:
            cv2.circle(transparent_layer, track, radius, (255, 0, 0, 255), -1)
    transparent_layer = Image.fromarray(transparent_layer, 'RGBA')
    selected_point_map = Image.alpha_composite(transparent_background, transparent_layer)
    return tracking_points, trackings_input_label, selected_point_map

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _= cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    combined_images = []  
    mask_images = []      
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        plt.axis('off')
        combined_filename = f"combined_image_{i+1}.jpg"
        plt.savefig(combined_filename, format='jpg', bbox_inches='tight')
        combined_images.append(combined_filename)
        plt.close()
        mask_image = np.zeros_like(image, dtype=np.uint8)
        mask_layer = (mask > 0).astype(np.uint8) * 255
        for c in range(3):
            mask_image[:, :, c] = mask_layer
        mask_filename = f"mask_image_{i+1}.png"
        Image.fromarray(mask_image).save(mask_filename)
        mask_images.append(mask_filename)
    return combined_images, mask_images

def expand_contract_mask(mask, px, expand=True):
    kernel = np.ones((px, px), np.uint8)
    if expand:
        return cv2.dilate(mask, kernel, iterations=1)
    else:
        return cv2.erode(mask, kernel, iterations=1)

def feather_mask(mask, feather_size=10):
    feathered_mask = mask.copy()
    Feathered_region = mask > 0
    Feathered_region = cv2.dilate(Feathered_region.astype(np.uint8), np.ones((feather_size, feather_size), np.uint8), iterations=1)
    Feathered_region = Feathered_region & (~mask.astype(bool))
    
    for i in range(1, feather_size + 1):
        weight = i / (feather_size + 1)
        feathered_mask[Feathered_region] = feathered_mask[Feathered_region] * (1 - weight) + weight

    return feathered_mask

def process_mask(mask, expand_contract_px, expand, feathering_enabled, feather_size):
    if expand_contract_px > 0:
        mask = expand_contract_mask(mask, expand_contract_px, expand)
    if feathering_enabled:
        mask = feather_mask(mask, feather_size)
    return mask

@spaces.GPU()
def sam_process(input_image, checkpoint, tracking_points, trackings_input_label, expand_contract_px, expand, feathering_enabled, feather_size):
    image = Image.open(input_image)
    image = np.array(image.convert("RGB"))
    sam21_hfmap = {
        "tiny": "facebook/sam2.1-hiera-tiny",
        "small": "facebook/sam2.1-hiera-small",
        "base-plus": "facebook/sam2.1-hiera-base-plus",
        "large": "facebook/sam2.1-hiera-large",
    }
    # sam2_checkpoint, model_cfg = checkpoint_map[checkpoint]
    # Use CPU for both model and computations
    # sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
    predictor = SAM2ImagePredictor.from_pretrained(sam21_hfmap[checkpoint], device="cuda")

    # predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)
    input_point = np.array(tracking_points.value)
    input_label = np.array(trackings_input_label.value)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    processed_masks = []
    for mask in masks:
        processed_mask = process_mask(mask, expand_contract_px, expand, feathering_enabled, feather_size)
        processed_masks.append(processed_mask)
    results, mask_results = show_masks(image, processed_masks, scores,
                                     point_coords=input_point,
                                     input_labels=input_label,
                                     borders=True)
    return results[0], mask_results[0]

with gr.Blocks() as demo:
    first_frame_path = gr.State()
    tracking_points = gr.State([])
    trackings_input_label = gr.State([])
    with gr.Column():
        gr.Markdown("# SAM2 Image Predictor / Masking Assistant")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="input image", interactive=False, type="filepath", visible=False)                 
                points_map = gr.Image(label="points map", type="filepath", interactive=True)
                with gr.Row():
                    point_type = gr.Radio(label="point type", choices=["include", "exclude"], value="include")
                    clear_points_btn = gr.Button("Clear Points")
                checkpoint = gr.Dropdown(label="Checkpoint", choices=["tiny", "small", "base-plus", "large"], value="base-plus")
                with gr.Row():
                    expand_contract_px = gr.Slider(minimum=0, maximum=50, value=0, label="Expand/Contract (pixels)")
                    expand = gr.Radio(["Expand", "Contract"], value="Expand", label="Action")
                with gr.Row():
                    feathering_enabled = gr.Checkbox(value=False, label="Enable Feathering")
                    feather_size = gr.Slider(minimum=1, maximum=50, value=10, label="Feathering Size", visible=False)
                submit_btn = gr.Button("Submit")
            with gr.Column():
                output_result = gr.Image()
                output_result_mask = gr.Image()
    clear_points_btn.click(
        fn=preprocess_image,
        inputs=input_image,
        outputs=[first_frame_path, tracking_points, trackings_input_label, points_map],
        queue=False
    )
    points_map.upload(
        fn=preprocess_image,
        inputs=[points_map],
        outputs=[first_frame_path, tracking_points, trackings_input_label, input_image],
        queue=False
    )
    points_map.select(
        fn=get_point,
        inputs=[point_type, tracking_points, trackings_input_label, first_frame_path],
        outputs=[tracking_points, trackings_input_label, points_map],
        queue=False
    )
    submit_btn.click(
        fn=sam_process,
        inputs=[input_image, checkpoint, tracking_points, trackings_input_label, expand_contract_px, expand, feathering_enabled, feather_size],
        outputs=[output_result, output_result_mask]
    )
    feathering_enabled.change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=[feathering_enabled],
        outputs=[feather_size]
    )

demo.launch(show_error=True)