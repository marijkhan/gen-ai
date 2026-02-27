import base64
import requests

HF_MODEL_URL = "https://router.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"


def generate_image(topic: str) -> tuple[str | None, str | None]:
    """Generate a professional LinkedIn banner image using Stable Diffusion XL via HuggingFace Inference API.

    Returns (base64_str, error). One of them will be None.
    """
    prompt = (
        f"Professional, modern LinkedIn banner image about: {topic}. "
        f"Clean design with abstract visuals, corporate color palette (blues, whites, grays). "
        f"No text, no words, no letters in the image. Suitable as a LinkedIn post header."
    )
    try:
        response = requests.post(
            HF_MODEL_URL,
            json={"inputs": prompt},
            timeout=120,
        )
        if response.status_code == 200:
            image_bytes = response.content
            return base64.b64encode(image_bytes).decode("utf-8"), None
        else:
            return None, f"HuggingFace API error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)
