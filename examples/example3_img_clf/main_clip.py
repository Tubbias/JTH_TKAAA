import argparse
import os

import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image


def main():
    parser = argparse.ArgumentParser(description='CLIP Image Classification example')
    parser.add_argument('--use-animals', action='store_true', help='Use animal images and prompts')
    args = parser.parse_args()

    if args.use_animals:
        image_paths = [
            'data/3_cat.png',
            'data/3_dog.png',
            'data/3_zebra.png'
        ]
        text_prompts = [
            'a photo of a cat',
            'a photo of a dog',
            'a photo of a zebra'
        ]
    else:
        image_paths = [
            'data/3_apple_golden_delicious.png',
            'data/3_apple_royal_gala.png',
            'data/3_bananas.png',
            'data/3_bell_pepper.png',
            'data/3_pineapple.png'
        ]
        text_prompts = [
            'a photo of golden delicious apples',
            #'a photo of green apples',
            'a photo of royal gala apples',
            'a photo of bananas',
            'a photo of a bell pepper',
            'a photo of a pineapple'
        ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = 'openai/clip-vit-large-patch14'

    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    images = [load_image(p) for p in image_paths]

    with torch.no_grad():
        inputs = processor(images=images, text=text_prompts, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        similarity = outputs.logits_per_image

        probs = similarity.softmax(dim=-1)

    correct = 0
    for row, img_path in enumerate(image_paths):
        top1_idx = int(similarity[row].argmax().item())
        correct += int(top1_idx == row)
        print(f"{os.path.basename(img_path)} → {text_prompts[top1_idx]} (p={probs[row, top1_idx].item():.3f})")

    print(f"\nAccuracy: {correct}/{len(image_paths)} = {correct/len(image_paths):.2%}")


if __name__ == "__main__":
    main()
