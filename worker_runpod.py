import os, json, requests, random, runpod

import numpy as np
from PIL import Image
import torch

from Marigold.marigold import MarigoldPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

with torch.inference_mode():
    normals_path = "/content/normals"
    dtype = torch.float32
    variant = None
    unet_normals         = UNet2DConditionModel.from_pretrained(normals_path, subfolder="unet")   
    vae_normals          = AutoencoderKL.from_pretrained(normals_path, subfolder="vae")  
    text_encoder_normals = CLIPTextModel.from_pretrained(normals_path, subfolder="text_encoder")  
    tokenizer_normals    = CLIPTokenizer.from_pretrained(normals_path, subfolder="tokenizer") 
    scheduler_normals    = DDIMScheduler.from_pretrained(normals_path, timestep_spacing="trailing", subfolder="scheduler") 
    pipe_normals = MarigoldPipeline.from_pretrained(pretrained_model_name_or_path = normals_path,
                                            unet=unet_normals, 
                                            vae=vae_normals, 
                                            scheduler=scheduler_normals, 
                                            text_encoder=text_encoder_normals, 
                                            tokenizer=tokenizer_normals, 
                                            variant=variant, 
                                            torch_dtype=dtype, 
                                            )
    pipe_normals = pipe_normals.to('cuda')
    pipe_normals.unet.eval()

    depth_path = "/content/depth"
    unet_depth         = UNet2DConditionModel.from_pretrained(depth_path, subfolder="unet")   
    vae_depth          = AutoencoderKL.from_pretrained(depth_path, subfolder="vae")  
    text_encoder_depth = CLIPTextModel.from_pretrained(depth_path, subfolder="text_encoder")  
    tokenizer_depth    = CLIPTokenizer.from_pretrained(depth_path, subfolder="tokenizer") 
    scheduler_depth    = DDIMScheduler.from_pretrained(depth_path, timestep_spacing="trailing", subfolder="scheduler") 
    pipe_depth = MarigoldPipeline.from_pretrained(pretrained_model_name_or_path = depth_path,
                                            unet=unet_depth, 
                                            vae=vae_depth, 
                                            scheduler=scheduler_depth, 
                                            text_encoder=text_encoder_depth, 
                                            tokenizer=tokenizer_depth, 
                                            variant=variant, 
                                            torch_dtype=dtype, 
                                            )
    pipe_depth = pipe_depth.to('cuda')
    pipe_depth.unet.eval()

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    original_file_name = url.split('/')[-1]
    _, original_file_extension = os.path.splitext(original_file_name)
    file_path = os.path.join(save_dir, file_name + original_file_extension)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image_check']
    input_image = download_file(url=input_image, save_dir='/content', file_name='input_image_tost')
    image = Image.open(input_image).convert('RGB')
    image_array = np.array(image).astype('uint8')
    pil_image = Image.fromarray(image_array)
    processing_res_choice = 768

    with torch.no_grad():
        pipe_out_normals = pipe_normals(pil_image, denoising_steps=1, ensemble_size=1, noise="zeros", normals=True, processing_res=processing_res_choice, match_input_res=True)
        pipe_out_depth = pipe_depth(pil_image, denoising_steps=1, ensemble_size=1, noise="zeros", normals=False, processing_res=processing_res_choice, match_input_res=True)

    pipe_out_normals.normal_colored.save("/content/marigold-normal-tost.png")
    pipe_out_depth.depth_colored.save("/content/marigold-depth-tost.png")

    result = ["/content/marigold-depth-tost.png", ["/content/marigold-normal-tost.png"]]
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result[0])
        with open(result[0], "rb") as file:
            files = {default_filename: file.read()}
        for path in result[1]:
            filename = os.path.basename(path)
            with open(path, "rb") as file:
                files[filename] = file.read()
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_urls = [attachment['url'] for attachment in response.json()['attachments']]
        notify_payload = {"jobId": job_id, "result": str(result_urls), "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": str(result_urls), "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists("/content/marigold-depth-tost.png"):
            os.remove("/content/marigold-depth-tost.png")
        if os.path.exists("/content/marigold-normal-tost.png"):
            os.remove("/content/marigold-normal-tost.png")

runpod.serverless.start({"handler": generate})