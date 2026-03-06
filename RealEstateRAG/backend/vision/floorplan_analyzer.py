from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class FloorplanAnalyzer:
    def __init__(self):
        # We load a small vision model suitable for CPU processing.
        # BLIP (Bootstrapping Language-Image Pre-training) is excellent for general captioning and visual QA.
        self.processor = None
        self.model = None

    def _load_model(self):
        # Lazy loading to prevent startup delays if vision isn't immediately used
        if self.processor is None:
            model_id = "Salesforce/blip-image-captioning-base"
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(model_id)

    def analyze_image(self, image: Image.Image) -> str:
        """
        Takes a PIL image of a floorplan or property photo and generates a descriptive caption.
        """
        self._load_model()
        
        # Pre-prompt to guide the BLIP model slightly towards architecture/real estate
        text = "a photograph of a real estate"
        
        inputs = self.processor(image, text, return_tensors="pt")
        
        out = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption

analyzer = FloorplanAnalyzer()
