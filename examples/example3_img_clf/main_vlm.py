import argparse

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

VLM_QWEN3_VL_2B = 'Qwen/Qwen3-VL-2B-Instruct'


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
        generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=0.3, repetition_penalty=1.2)
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)

        print(decoded)
        return decoded


def main():
    parser = argparse.ArgumentParser(description='CLIP Image Classification example')
    parser.add_argument('--use-animals', action='store_true', help='Use animal images and prompts')
    args = parser.parse_args()

    image_query = "What is shown in the image, describe both the content and color? Be short!"

    if args.use_animals:
        image_paths = [
            'data/3_cat.png',
            'data/3_dog.png',
            'data/3_zebra.png'
        ]
        gt_class = [
            ['cat'],
            ['dog'],
            ['zebra']
        ]
    else:
        image_paths = [
            'data/3_apple_golden_delicious.png',
            'data/3_apple_royal_gala.png',
            'data/3_bananas.png',
            'data/3_bell_pepper.png',
            'data/3_pineapple.png'
        ]
        gt_class = [
            ['green', 'apple'],
            ['red', 'apple'],
            ['bananas'],
            ['bell', 'pepper'],
            ['pineapple']
        ]

    model_type = VLM_QWEN3_VL_2B

    vlm_processor = VLMQwen3(model_type=model_type)
    class_correct_cnt = 0
    for idx, image_path in enumerate(image_paths):
        print(f"Processing image: {image_path}")
        response = vlm_processor.process(image_path=image_path, image_query=image_query)
        gt_class_entry = gt_class[idx]

        clf_correct = True
        for gt_class_property in gt_class_entry:
            if gt_class_property.lower() not in response.lower():
                clf_correct = False
                break

        if clf_correct:
            class_correct_cnt += 1
            print(f"Correctly identified as {gt_class[idx]}")
        else:
            print(f"Incorrectly identified. GT: {gt_class[idx]} | Response: {response}")

    print(f"\nFinal Accuracy: {class_correct_cnt}/{len(image_paths)} = {class_correct_cnt/len(image_paths):.2%}")


if __name__ == "__main__":
    main()
