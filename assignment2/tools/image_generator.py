import io
import base64
from huggingface_hub import InferenceClient
from config import HF_TOKEN

_client = InferenceClient(
    provider="nscale",
    api_key=HF_TOKEN,
)


def generate_image(topic: str) -> tuple[str | None, str | None]:
    """Generate a professional LinkedIn banner image using Stable Diffusion XL via HuggingFace Inference Client.

    Returns (base64_str, error). One of them will be None.
    """
    prompt = (
        f"Professional, modern LinkedIn banner image about: {topic}. "
        f"Clean design with abstract visuals, corporate color palette (blues, whites, grays). "
        f"No text, no words, no letters in the image. Suitable as a LinkedIn post header."
    )
    try:
        image = _client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
        )
        # image is a PIL.Image object — convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_b64, None
    except Exception as e:
        return None, str(e)
