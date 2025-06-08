# ==========================================================================================
#
# MIT License. To view a copy of the license, visit MIT_LICENSE.md.
#
# ==========================================================================================

import argparse
import sys
import os
import numpy as np
import torch
import json
from PIL import Image

sys.path.append('./')
from src.diffusers_model_pipeline import LaTexBlendXLPipeline, MultiConceptprocessor

def sample(ckpt, from_file, compress, deep_replace, freeze_model, concept_list, sdxl=False, device="cuda:0", outdir=None, seed=None):
    with open(concept_list, "r") as f:
        concept_list = json.load(f)[0]

    model_id = ckpt
    if sdxl:
        pipe = LaTexBlendXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, stage="inference", get_concept_bank=concept_list['get_concept_bank']).to(device)

    pipe.load_model(concept_list, compress, deep_replace)

    need_name = False
    if outdir == None:
        need_name = True
        outdir = os.path.dirname(concept_list['concept1']["delta_ckpt"])
    else:
        outdir = outdir.replace('.', '').replace(',', '') + '/'
        os.makedirs(outdir, exist_ok=True)
    
    if seed is None: 
        generator = torch.Generator(device=device).manual_seed(42)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)

    all_images = []
    if from_file is None:
        prompt = [concept_list['prompt']]*concept_list['batch_size']
        images = pipe(prompt, num_inference_steps=concept_list['num_inference_steps'], guidance_scale=6., eta=1., generator=generator, concept_list=concept_list, stage='inference').images
        all_images += images
        images = np.hstack([np.array(x) for x in images])
        images = Image.fromarray(images)

        name = '-'.join(prompt[0][:50].split())
        if need_name:
            images.save(f'{outdir}/{name}.png')
        else:
            images.save(f'{outdir}/all.png')
    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = [[prompt]*concept_list['batch_size'] for prompt in data]

        for prompt in data:
            images = pipe(prompt, num_inference_steps=concep_list['num_inference_steps'], guidance_scale=6., eta=1., generator=generator, concept_list=concept_list, stage='inference').images
            all_images += images
            images = np.hstack([np.array(x) for x in images], 0)
            images = Image.fromarray(images)
  
            name = '-'.join(prompt[0][:50].split())
            images.save(f'{outdir}/{name}.png')

    os.makedirs(f'{outdir}/samples', exist_ok=True)
    base_count = len(os.listdir(f'{outdir}/samples'))
    for i, im in enumerate(all_images):
        im.save(f'{outdir}/samples/{i+base_count}.jpg')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ckpt', help='target string for query',
                        type=str)
    parser.add_argument('--delta_ckpt', help='first target string for query', default=None,
                        type=str)
    parser.add_argument('--modifier_tokens', help='modifier token', default=None,
                        type=str)
    parser.add_argument("--deep_replace", action='store_true')
    parser.add_argument('--from-file', help='path to prompt file', default=None,
                        type=str)
    parser.add_argument('--prompt', help='prompt to generate', default=None,
                        type=str)
    parser.add_argument("--compress", action='store_true')
    parser.add_argument("--sdxl", action='store_true')
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv',
                        type=str)
    parser.add_argument('--device', help='device to use', default='cuda:0',
                        type=str)
    parser.add_argument('--output_file', help='filename of output', 
                        type=str)  
    parser.add_argument('--seed', help='seed for sample', default=None,
                        type=int)         

    parser.add_argument(
        "--concept_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.from_file, args.compress, args.deep_replace, args.freeze_model, args.concept_list, args.sdxl, args.device, args.output_file, args.seed)
