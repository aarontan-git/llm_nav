import requests
import base64


API_KEY = "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"

def asking_sam_altman(image_path, text_prompt, model="gpt-4-turbo"):
    """
    Queries GPT-4 Vision with an image (if provided) and a text prompt, and returns the content of the response.

    Parameters:
    - image_path (str, optional): Path to the image file. If None, no image will be sent.
    - text_prompt (str): Text prompt for the query.
    - model (str): The model to use for the query.

    Returns:
    - str: The content from the JSON response of the API.
    """
    import os

    content = [{"type": "text", "text": text_prompt}]

    if image_path and os.path.exists(image_path):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        encoded_image = encode_image(image_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}", "detail": "high"}})
    elif image_path:
        print("Image path does not exist:", image_path)
        return "Image path does not exist."

    # API setup
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 4096
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # Send the query
    response = requests.post(url=url, headers=headers, json=payload)
    response_json = response.json()

    # Extract the 'content' from the response
    content_response = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')

    return content_response