import argparse

from io import BytesIO
import os

from google import genai
from google.genai.types import Content, Part
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VLM_QWEN3_VL_2B = 'Qwen/Qwen3-VL-2B-Instruct'

# LaTeX recipe template for cocktail recipes collected from https://www.overleaf.com/latex/templates/latex-cookbook-modular-latex-cookbook-template/xxhmjsbxbdyg
latex_template_recipe = (''
                         '\setRecipeMeta{RECEIPE_NAME}{}{}{}{IMAGE_PATH})'
                         '"\"begin{recipe}'
                         '"\"begin{ingredients}'
                         '\ingredient{INGREDIENT_1}'
                         '\ingredient{INGREDIENT_2}'
                         '\end{ingredients}'
                         '"\"begin{steps}'
                         '\step{STEP_1_DESCRIPTION}'
                         '\step{STEP_2_DESCRIPTION}'
                         '\end{steps}'
                         '\end{recipe}')

class Gemini:
    def __init__(self, model_type):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_type = model_type

    def process(self, image_path, image_query):
        img_bytes = open(image_path, "rb").read()
        resp = self.client.models.generate_content(
            model=self.model_type,
            contents=Content(
                role="user",
                parts=[
                    Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    Part.from_text(text=image_query)
                ]
            )
        )
        return resp.text.strip()

    def process_text(self, text_query):
        resp = self.client.models.generate_content(
            model=self.model_type,
            contents=Content(
                role="user",
                parts=[
                    Part.from_text(text=text_query)
                ]
            )
        )
        return resp.text.strip()



class VLMImageGenerator:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate_image(self, text_prompt):
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[text_prompt],
        )

        image = None
        text = None
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                text = part
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))

        return image, text


class VLMQwen3:
    def __init__(self, model_type):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_type, device_map="auto", dtype='auto').eval()
        self.processor = AutoProcessor.from_pretrained(model_type)

    def process(self, image_path, image_query):
        image_query_chat = image_query.replace("<image>", "").strip()

        system_prompt = 'You are a helpful assistant.'
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            },
            {
                "role": "user", "content": [
                {"type": "image",
                 "url": image_path},
                {"type": "text", "text": image_query_chat},
            ]
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]
        generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=0.3, top_p=0.7, repetition_penalty=1.2)
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)

        print(decoded)
        return decoded

    def process_text(self, text_query):
        inputs = self.processor(
            text=text_query,
            return_tensors="pt",
            padding=True
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generation = self.model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.3, top_p=0.7, repetition_penalty=1.2)
        decoded = self.processor.decode(generation[0], skip_special_tokens=True)

        print(decoded)
        return decoded


def main():
    parser = argparse.ArgumentParser(description='CLIP Image Classification example')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input drink table image')
    parser.add_argument('--output-folder', type=str, required=True, help='Folder to save the results')
    parser.add_argument('--use-local-vlm', action='store_true', help='Use local VLM model instead of Gemini API')
    parser.add_argument('--use-ocr-vlm-result', action='store_true', help='Use OCR result from VLM for recipe generation')
    args = parser.parse_args()

    # Initialize VLM processor
    if args.use_local_vlm:
        vlm_processor = VLMQwen3(model_type=VLM_QWEN3_VL_2B)
    else:
        model_type = "gemini-2.5-pro"
        #model_type = "gemini-2.5-flash"
        vlm_processor = Gemini(model_type=model_type)

    # Analyze input image to get liquor types/brands, either via OCR or direct recognition
    if args.use_ocr_vlm_result:
        image_query_vlm_ocr = "Read the OCR in the image?"
        response_vlm_ocr = vlm_processor.process(image_path=args.image_path, image_query=image_query_vlm_ocr)
        #print(response_vlm_ocr)

        image_query_vlm_ocr_res = (f'Based on this text: "{response_vlm_ocr}", what type of liquor types/brands can you '
                                   f'find? Output ONLY a short list of the resulting types/brands in json.')
        response_vlm_ocr_formatted = vlm_processor.process_text(text_query=image_query_vlm_ocr_res)
        print(response_vlm_ocr_formatted)

        image_query_recipe = response_vlm_ocr_formatted
    else:
        image_query_vlm = "What type of liquors are you seeing in the image? Provide JUST a short structured list in json."
        response_vlm = vlm_processor.process(image_path=args.image_path, image_query=image_query_vlm)
        print(response_vlm)

        image_query_recipe = response_vlm

    # 2. Generate cocktail recipes based on liquor types/brands
    #image_query_cocktail_recipe = (f"Based on the provided list of liquor types/brands f{image_query_recipe}, can you "
    #                               f"suggest 5 cocktail recipes that can be made using these liquors? Assume that I "
    #                               f"have standard drink accessories. Only provide plain recipe text of each drink "
    #                               f"while keeping them separated by a #.")
    image_query_cocktail_recipe = (f"Based on the provided list of liquor types/brands f{image_query_recipe}, can you "
                                   f"suggest 5 cocktail recipes that can be made using these liquors? Assume that I "
                                   f"have standard drink accessories and be VERY creative. ONLY (no extra information) provide plain recipe text of each drink "
                                   f"while you MUST keeping them separated by a #.")
    response_cocktail_recipes = vlm_processor.process_text(text_query=image_query_cocktail_recipe)
    print(response_cocktail_recipes)

    # 3. For each cocktail recipe, generate an image and LaTeX recipe file
    image_generator = VLMImageGenerator()
    cocktail_drinks = response_cocktail_recipes.split('#')
    idx = 0
    for drink_text in cocktail_drinks:
        image_query_text = f'Generate a high quality image of the cocktail drink based on this description: {drink_text.strip()}'
        #image_query_text = f'Generate an image of the cocktail drink based on this description in a student dorm environment: {drink_text.strip()}'
        drink_image, drink_text = image_generator.generate_image(text_prompt=image_query_text)

        if not drink_image or not drink_text:
            print('Skipping drink due to missing image or text')
            continue

        if args.use_local_vlm:
            # Qwen3 cannot consistently generate good drink names (not separating the recipe output as good), so use a default name
            image_name = 'drink'

            # Check if the generated text is a structured recipe
            is_recipe_response = vlm_processor.process_text(text_query=f'Is this really a structured drink recipe with ingredients and steps: {drink_text.text}. ONLY answer with YES or NO.')
            if 'yes' not in is_recipe_response.lower():
                print('Skipping drink due to invalid recipe format')
                continue
        else:
            image_name = vlm_processor.process_text(text_query=f'Provide a short name for this drink: {drink_text.text}. If multiple words, '
                                                               f'connect them with underscores. Provide ONLY the name as output.')

        drink_image_path = f'{args.output_folder}/{image_name}_{idx}.png'
        drink_image.save(drink_image_path)

        latex_query = (f'Using LaTeX, provide a full cocktail recipe in LaTeX format as follows: {latex_template_recipe} '
                       f'based on this description: {drink_text.text}. Provide ONLY the LaTeX code as output.')
        latex_recipe_code = vlm_processor.process_text(text_query=latex_query)
        latex_file_path = f'{args.output_folder}/{image_name}_{idx}.tex'
        with open(latex_file_path, 'w') as f:
            f.write(latex_recipe_code)

        idx += 1


if __name__ == "__main__":
    main()
