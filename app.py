import spaces
import gradio as gr
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from PIL import Image, ImageDraw
import numpy as np
# from gradio.sketch.run import create

MODELS = {
    "RealVisXL V5.0 Lightning": "SG161222/RealVisXL_V5.0_Lightning",
    "Lustify Lightning": "GraydientPlatformAPI/lustify-lightning",
    "Juggernaut XL Lightning": "RunDiffusion/Juggernaut-XL-Lightning",
    "Juggernaut-XL-V9-GE-RDPhoto2": "AiWise/Juggernaut-XL-V9-GE-RDPhoto2-Lightning_4S",
    "SatPony-Lightning": "John6666/satpony-lightning-v2-sdxl"
}

config_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="config_promax.json",
)
config = ControlNetModel_Union.load_config(config_file)
controlnet_model = ControlNetModel_Union.from_config(config)
model_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model_promax.safetensors",
)
state_dict = load_state_dict(model_file)
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)
model.to(device="cuda", dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to("cuda")
pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    torch_dtype=torch.float16,
    vae=vae,
    controlnet=model,
    variant="fp16",
)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
print(pipe)

def load_default_pipeline():
    global pipe
    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=torch.float16,
        vae=vae,
        controlnet=model,
    ).to("cuda")
    return gr.update(value="Default pipeline loaded!")

@spaces.GPU(duration=7)
def fill_image(prompt, image, model_selection, paste_back):
    print(f"Received image: {image}")
    if image is None:
        yield None, None
        return

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(prompt, "cuda", True)
    source = image["background"]
    mask = image["layers"][0]
    alpha_channel = mask.split()[3]
    binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
    cnet_image = source.copy()
    cnet_image.paste(0, (0, 0), binary_mask)

    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
    ):
        yield image, cnet_image

    print(f"{model_selection=}")
    print(f"{paste_back=}")
    if paste_back:
        image = image.convert("RGBA")
        cnet_image.paste(image, (0, 0), binary_mask)
    else:
        cnet_image = image
    yield source, cnet_image

def clear_result():
    return gr.update(value=None)

def can_expand(source_width, source_height, target_width, target_height, alignment):
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    target_size = (width, height)
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    source = image.resize((new_width, new_height), Image.LANCZOS)
    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "80%":
        resize_percentage = 80
    elif resize_option == "66%":
        resize_percentage = 66
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:  # Custom
        resize_percentage = custom_resize_percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)
    source = source.resize((new_width, new_height), Image.LANCZOS)
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)
    white_gaps_patch = 2
    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        botttom_overlap = margin = margin = margin = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height
    mask_draw.rectangle([
        (left_overlap, top_overlap),
        (right_overlap, bottom_overlap)
    ], fill=0)
    return background, mask
def preview_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    preview = background.copy().convert('RGBA')
    red_overlay = Image.new('RGBA', background.size, (255, 0, 0, 64))
    red_mask = Image.new('RGBA', background.size, (0, 0, 0, 0))
    red_mask.paste(red_overlay, (0, 0), mask)
    preview = Image.alpha_composite(preview, red_mask)
    return preview

@spaces.GPU(duration=12)
def inpaint(prompt, image, inpaint_model, paste_back):
    global pipe
    if pipe.config.model_name != MODELS[model_name]:
        pipe = StableDiffusionXLFillPipeline.from_pretrained(
            MODELS[model_name],
            torch_dtype=torch.float16,
            vae=vae,
            controlnet=model,
        ).to("cuda")
        print(f"Loaded new SDXL model: {pipe.config.model_name}")
    mask = Image.fromarray(image["mask"]).convert("L")
    image = Image.fromarray(image["image"])
    inpaint_final_prompt = f"score_9, score_8_up, score_7_up, {prompt}"
    result = pipe(prompt=inpaint_final_prompt, image=image, mask_image=mask).images[0]
    if paste_back:
        result.paste(image, (0, 0), Image.fromarray(255 - np.array(mask)))
    return result

@spaces.GPU(duration=12)
def outpaint(image, width, height, overlap_percentage, num_inference_steps, resize_option, custom_resize_percentage, prompt_input, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    if not can_expand(background.width, background.height, width, height, alignment):
        alignment = "Middle"
    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)
    final_prompt = f"score_9, score_8_up, score_7_up, {prompt_input} , high quality, 4k"
    print(f"Outpainting using SDXL model: {pipe.config.model_name}")
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(final_prompt, "cuda", True)
    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=num_inference_steps
    ):
        yield cnet_image, image
    image = image.convert("RGBA")
    cnet_image.paste(image, (0, 0), mask)
    yield background, cnet_image

@spaces.GPU(duration=12)
def infer(image, width, height, overlap_percentage, num_inference_steps, resize_option, custom_resize_percentage, prompt_input, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    if not can_expand(background.width, background.height, width, height, alignment):
        alignment = "Middle"
    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)
    final_prompt = f"{prompt_input} , high quality, 4k"
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(final_prompt, "cuda", True)
    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=num_inference_steps
    ):
        yield cnet_image, image
    image = image.convert("RGBA")
    cnet_image.paste(image, (0, 0), mask)
    yield background, cnet_image

def use_output_as_input(output_image):
    return gr.update(value=output_image[1])

def preload_presets(target_ratio, ui_width, ui_height):
    if target_ratio == "9:16":
        changed_width = 768
        changed_height = 1280
        return changed_width, changed_height, gr.update()
    elif target_ratio == "2:3":
        changed_width = 1024
        changed_height = 1536
        return changed_width, changed_height, gr.update()
    elif target_ratio == "16:9":
        changed_width = 1280
        changed_height = 768
        return changed_width, changed_height, gr.update()
    elif target_ratio == "1:1":
        changed_width = 1024
        changed_height = 1024
        return changed_width, changed_height, gr.update()
    elif target_ratio == "Custom":
        return ui_width, ui_height, gr.update(open=True)
    else:
        return ui_width, ui_height, gr.update()

def select_the_right_preset(user_width, user_height):
    if user_width == 720 and user_height == 1280:
        return "9:16"
    elif user_width == 1024 and user_height == 1536:
        return "2:3"
    elif user_width == 1280 and user_height == 720:
        return "16:9"
    elif user_width == 1024 and user_height == 1024:
        return "1:1"
    else:
        return "Custom"

def toggle_custom_resize_slider(resize_option):
    return gr.update(visible=(resize_option == "Custom"))

def update_history(new_image, history):
    if history is None:
        history = []
    history.insert(0, new_image)
    return history

def clear_cache():
    global pipe
    pipe = None
    torch.cuda.empty_cache()
    return gr.update(value="Cache cleared!")

css = """
.nulgradio-container {
    width: 86vw !important;
}
.nulcontain {
    overflow-y: scroll !important;
    padding: 10px 40px !important;
}
div#component-17 {
    height: auto !important;
}


@media screen and (max-width: 600px) {
    .img-row{
        display: block !important;
        margin-bottom: 20px !important;
    }
    div#component-16 {
        display: block !important;
    }
}

"""

title = """<h1 align="center">Diffusers Image Outpaint</h1>
<div align="center">Drop an image you would like to extend, pick your expected ratio and hit Generate.</div>
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
    <p style="display: flex;gap: 6px;">
         <a href="https://huggingface.co/spaces/fffiloni/diffusers-image-outpout?duplicate=true">
            <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-md.svg" alt="Duplicate this Space">
        </a> to skip the queue and enjoy faster inference on the GPU of your choice
    </p>
</div>
"""

# with gr.Blocks() as sketchapp:
#     sketchapp = create("test.py", "test.py.json")
    
    
with gr.Blocks(css=css, fill_height=True) as demo:
    gr.Markdown("# Diffusers Inpaint and Outpaint")
    with gr.Tabs():
        with gr.TabItem("Inpaint"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(
                            label="Prompt",
                            info="Describe what to inpaint the mask with",
                            lines=3,
                        )
                    with gr.Column():
                        model_selection = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="RealVisXL V5.0 Lightning",
                            label="Model",
                        )
                        with gr.Row():
                            run_button = gr.Button("Generate")
                            paste_back = gr.Checkbox(True, label="Paste back original")
                with gr.Row(equal_height=False):
                    input_image = gr.ImageMask(
                        type="pil", label="Input Image", layers=True, elem_classes="img-row"
                    )
                    result = ImageSlider(
                        interactive=False,
                        label="Generated Image", 
                        elem_classes="img-row"
                    )
                use_as_input_button = gr.Button("Use as Input Image", visible=False)
                use_as_input_button.click(
                    fn=use_output_as_input, inputs=[result], outputs=[input_image]
                )
                run_button.click(
                    fn=clear_result,
                    inputs=None,
                    outputs=result,
                ).then(
                    fn=lambda: gr.update(visible=False),
                    inputs=None,
                    outputs=use_as_input_button,
                ).then(
                    fn=fill_image,
                    inputs=[prompt, input_image, model_selection, paste_back],
                    outputs=[result],
                ).then(
                    fn=lambda: gr.update(visible=True),
                    inputs=None,
                    outputs=use_as_input_button,
                )
                prompt.submit(
                    fn=clear_result,
                    inputs=None,
                    outputs=result,
                ).then(
                    fn=lambda: gr.update(visible=False),
                    inputs=None,
                    outputs=use_as_input_button,
                ).then(
                    fn=fill_image,
                    inputs=[prompt, input_image, model_selection, paste_back],
                    outputs=[result],
                ).then(
                    fn=lambda: gr.update(visible=True),
                    inputs=None,
                    outputs=use_as_input_button,
                )
        with gr.TabItem("Outpaint"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        input_image_outpaint = gr.Image(
                            type="pil",
                            label="Input Image"
                        )
                        with gr.Row():
                            with gr.Column(scale=2):
                                prompt_input = gr.Textbox(label="Prompt (Optional)")
                            with gr.Column(scale=1):
                                runout_button = gr.Button("Generate")
                        with gr.Row():
                            target_ratio = gr.Radio(
                                label="Expected Ratio",
                                choices=["2:3", "9:16", "16:9", "1:1", "Custom"],
                                value="1:1",
                                scale=2
                            )
                            alignment_dropdown = gr.Dropdown(
                                choices=["Middle", "Left", "Right", "Top", "Bottom"],
                                value="Middle",
                                label="Alignment"
                            )
                        with gr.Accordion(label="Advanced settings", open=False) as settings_panel:
                            with gr.Column():
                                with gr.Row():
                                    width_slider = gr.Slider(
                                        label="Target Width",
                                        minimum=720,
                                        maximum=1536,
                                        step=8,
                                        value=1024,
                                    )
                                    height_slider = gr.Slider(
                                        label="Target Height",
                                        minimum=720,
                                        maximum=1536,
                                        step=8,
                                        value=1024,
                                    )
                                num_inference_steps = gr.Slider(label="Steps", minimum=4, maximum=12, step=1, value=8)
                                with gr.Group():
                                    overlap_percentage = gr.Slider(
                                        label="Mask overlap (%)",
                                        minimum=1,
                                        maximum=80,
                                        value=10,
                                        step=1
                                    )
                                    with gr.Row():
                                        overlap_top = gr.Checkbox(label="Overlap Top", value=True)
                                        overlap_right = gr.Checkbox(label="Overlap Right", value=True)
                                    with gr.Row():
                                        overlap_left = gr.Checkbox(label="Overlap Left", value=True)
                                        overlap_bottom = gr.Checkbox(label="Overlap Bottom", value=True)
                                with gr.Row():
                                    resize_option = gr.Radio(
                                        label="Resize input image",
                                        choices=["Full", "80%", "66%", "50%", "33%", "25%", "Custom"],
                                        value="Full"
                                    )
                                    custom_resize_percentage = gr.Slider(
                                        label="Custom resize (%)",
                                        minimum=1,
                                        maximum=100,
                                        step=1,
                                        value=50,
                                        visible=False
                                    )
                                with gr.Column():
                                    preview_button = gr.Button("Preview alignment and mask")
                        # gr.Examples(
                        #     examples=[
                        #         ["./examples/example_1.webp", 1280, 720, "Middle"],
                        #         ["./examples/example_2.jpg", 1440, 810, "Left"],
                        #         ["./examples/example_3.jpg", 1024, 1024, "Top"],
                        #         ["./examples/example_3.jpg", 1024, 1024, "Bottom"],
                        #     ],
                        #     inputs=[input_image_outpaint, width_slider, height_slider, alignment_dropdown],
                        # )
                    with gr.Column():
                        result_outpaint = ImageSlider(
                            interactive=False,
                            label="Generated Image",
                        )
                        use_as_input_button_outpaint = gr.Button("Use as Input Image", visible=False)
                        history_gallery = gr.Gallery(label="History", columns=6, object_fit="contain", interactive=False)
                        preview_image = gr.Image(label="Preview")
        with gr.TabItem("Misc"):
            with gr.Column():
                clear_cache_button = gr.Button("Clear CUDA Cache")
                clear_cache_message = gr.Markdown("")
                clear_cache_button.click(
                    fn=clear_cache,
                    inputs=None,
                    outputs=clear_cache_message,
                )
                load_default_button = gr.Button("Load Default Pipeline")
                load_default_message = gr.Markdown("")
                load_default_button.click(
                    fn=load_default_pipeline,
                    inputs=None,
                    outputs=load_default_message,
                )
        # with gr.TabItem("Sketch"):
        #     sketchapp.render()   
                
    target_ratio.change(
        fn=preload_presets,
        inputs=[target_ratio, width_slider, height_slider],
        outputs=[width_slider, height_slider, settings_panel],
        queue=False
    )

    width_slider.change(
        fn=select_the_right_preset,
        inputs=[width_slider, height_slider],
        outputs=[target_ratio],
        queue=False
    )

    height_slider.change(
        fn=select_the_right_preset,
        inputs=[width_slider, height_slider],
        outputs=[target_ratio],
        queue=False
    )

    resize_option.change(
        fn=toggle_custom_resize_slider,
        inputs=[resize_option],
        outputs=[custom_resize_percentage],
        queue=False
    )
                        
    use_as_input_button_outpaint.click(
        fn=use_output_as_input,
        inputs=[result_outpaint],
        outputs=[input_image_outpaint]
    )
    runout_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result_outpaint,
    ).then(
        fn=infer,
        inputs=[input_image_outpaint, width_slider, height_slider, overlap_percentage, num_inference_steps,
                resize_option, custom_resize_percentage, prompt_input, alignment_dropdown,
                overlap_left, overlap_right, overlap_top, overlap_bottom],
        outputs=[result_outpaint],
    ).then(
        fn=lambda x, history: update_history(x[1], history),
        inputs=[result_outpaint, history_gallery],
        outputs=history_gallery,
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=[use_as_input_button_outpaint],
    )
    prompt_input.submit(
        fn=clear_result,
        inputs=None,
        outputs=result_outpaint,
    ).then(
        fn=infer,
        inputs=[input_image_outpaint, width_slider, height_slider, overlap_percentage, num_inference_steps,
                resize_option, custom_resize_percentage, prompt_input, alignment_dropdown,
                overlap_left, overlap_right, overlap_top, overlap_bottom],
        outputs=[result_outpaint],
    ).then(
        fn=lambda x, history: update_history(x[1], history),
        inputs=[result_outpaint, history_gallery],
        outputs=history_gallery,
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=[use_as_input_button_outpaint],
    )
    preview_button.click(
        fn=preview_image_and_mask,
        inputs=[input_image_outpaint, width_slider, height_slider, overlap_percentage, resize_option, custom_resize_percentage, alignment_dropdown,
                overlap_left, overlap_right, overlap_top, overlap_bottom],
        outputs=[preview_image],
        queue=False
    )

demo.launch(show_error=True)