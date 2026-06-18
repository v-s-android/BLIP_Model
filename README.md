Hugging Face offers a platform to experiment with BLIP and other AI models

Ensure you have Python and Transformers library installed. If not, you can install the transformers library using pip. Refer to the following code.

prerequisite:
```
pip3 install virtualenv 
virtualenv my_env # create a virtual environment my_env
source my_env/bin/activate # activate my_env
```
and install
```
pip install langchain==0.1.11 gradio==5.23.2 transformers==4.38.2 bs4==0.0.2 requests==2.31.0 torch==2.2.1
```

```
# Install the transformers library
!pip install transformers Pillow torch torchvision torchaudio
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image
image = Image.open("path_to_your_image.jpg")

# Prepare the image
inputs = processor(image, return_tensors="pt")

# Generate captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)
 
print("Generated Caption:", caption)
```

**Visual Question Answering**
```
# Install required libraries
!pip install transformers Pillow torch torchvision torchaudio

# Import required modules
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP processor and model (large version for better understanding + VQA)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load image from URL
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# Define question about the image
question = "What is in the image?"

# Prepare both image + question as input
inputs = processor(image, question, return_tensors="pt")

# Generate answer
outputs = model.generate(**inputs)

# Decode output into readable text
answer = processor.decode(outputs[0], skip_special_tokens=True)

# Print final answer
print("Answer:", answer)
```
