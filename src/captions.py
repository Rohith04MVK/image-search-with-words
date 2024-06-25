from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from .config import MODEL_NAME

class ImageCaptionGenerator:
    """A class to generate captions for images using a pre-trained BLIP model."""

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initializes the ImageCaptionGenerator with the given huggingface model.

        Args:
            model_name (str): The name of the model to use for image captioning. Default is "Salesforce/blip-image-captioning-large".
        """
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def generate_caption(self, image_path: str, text_prompt: str = "an image of", max_tokens: int = 16) -> str:
        """
        Generates a caption for the given image.

        Args:
            image_path (str): The path to the image file.
            text_prompt (str): The text prompt for conditional image captioning. Default is "an image of".
            max_tokens (int): The maximum number of tokens to generate. Default is 16.

        Returns:
            str: The generated caption.
        """
        # Load and preprocess the image
        raw_image = Image.open(image_path).convert('RGB')

        # Process the image and text
        inputs = self.processor(raw_image, text_prompt, return_tensors="pt")

        # Generate the caption
        out = self.model.generate(**inputs, max_new_tokens=max_tokens)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption


model = ImageCaptionGenerator()


def get_model():
    return model
