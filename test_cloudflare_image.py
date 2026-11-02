
import os
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
api_token = os.getenv("CLOUDFLARE_API_TOKEN")

if not account_id or not api_token:
    raise ValueError("Cloudflare credentials missing in .env")

model = "@cf/black-forest-labs/flux-1-schnell"

url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

payload = {
    "prompt": "A clear educational medical illustration of the human heart, showing four chambers and major blood vessels, anatomically accurate, white background"
}

print("Generating medical image...")

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=180
)

if not response.ok:
    print("Status:", response.status_code)
    print("Error:", response.text)
    raise SystemExit("Image generation failed")

result = response.json()

if not result.get("success"):
    print(result)
    raise SystemExit("Cloudflare returned an error")

image_data = result["result"]["image"]

output_dir = ROOT / "static" / "generated_images"
output_dir.mkdir(parents=True, exist_ok=True)

image_path = output_dir / "cloudflare_test.jpg"
image_path.write_bytes(base64.b64decode(image_data))

print("Image generated successfully!")
print("Saved at:", image_path)
