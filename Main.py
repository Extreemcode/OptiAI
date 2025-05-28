import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def describe_image(image_path):

    image = Image.open(image_path).convert('RGB')

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    inputs = processor(image, return_tensors="pt")

 
    with torch.no_grad():
        output = model.generate(**inputs)

   
    description = processor.decode(output[0], skip_special_tokens=True)
    return description

while True:
    image_choice = input('Would you like to use a preset image or a custom image? (No Capitals)')

    if image_choice == "custom":
        image_path = input("Enter the path to the file: ")
    if os.path.exists(image_path):  # Check if file exists
        description = describe_image(image_path)
        print('')
        print('')
        print('') 
        print('')
        print('')
        print(description)
        print('')
        print('')
        print('')
        print('')
        print('')
        
    elif image_choice == "preset":
        preset_image = input('Choose a preset image (1,2,3)')
        if preset_image in ["1", "2", "3"]:
            description = describe_image('/workspaces/codespaces-blank/_be32e89c-07ed-4e3b-9ecc-b0f8941b1b33.jpg')
            print('')
            print('')
            print('')
            print('')
            print('')
            print(description)
            print('')
            print('')
            print('')
            print('')
            print('')
            
        else:
            print("Invalid preset selection. Please choose 1, 2, or 3.")

    else:
        print("Invalid choice. Please enter 'custom' or 'preset'.")
