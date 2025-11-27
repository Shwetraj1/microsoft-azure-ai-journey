import os
import json

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
import requests


def main(): 
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')
        
    try: 
        # Load environment settings
        load_dotenv()
        endpoint = os.getenv("ENDPOINT")
        model_deployment = os.getenv("MODEL_DEPLOYMENT")
        api_version = os.getenv("API_VERSION")

        if not endpoint or not model_deployment or not api_version:
            raise Exception("Missing required environment variables.")

        # Initialize the Azure OpenAI client
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(
                exclude_environment_credential=True,
                exclude_managed_identity_credential=True
            ), 
            "https://cognitiveservices.azure.com/.default"
        )

        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider
        )

        img_no = 0

        # Loop for prompts
        while True:
            input_text = input("Enter the prompt (or type 'quit' to exit): ")

            if input_text.lower() == "quit":
                break

            if len(input_text.strip()) == 0:
                print("Please enter a prompt.")
                continue
            
            # Generate the image
            result = client.images.generate(
                model=model_deployment,
                prompt=input_text,
                n=1
            )

            json_response = json.loads(result.model_dump_json())
            image_url = json_response["data"][0]["url"]

            # Save the image
            img_no += 1
            file_name = f"image_{img_no}.png"
            save_image(image_url, file_name)

    except Exception as ex:
        print("Error:", ex)


def save_image(image_url, file_name):
    image_dir = os.path.join(os.getcwd(), 'images')

    # Create directory if missing
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    image_path = os.path.join(image_dir, file_name)

    # Download and save image
    generated_image = requests.get(image_url).content
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)

    print(f"Image saved as {image_path}")


if __name__ == '__main__': 
    main()
