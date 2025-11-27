import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import io
import tempfile
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from matplotlib import pyplot as plt




# Load environment variables
load_dotenv()
AI_ENDPOINT = os.getenv('AI_SERVICE_ENDPOINT')
AI_KEY = os.getenv('AI_SERVICE_KEY')




def analyze_image(image_data):
    client = ImageAnalysisClient(
        endpoint=AI_ENDPOINT,
        credential=AzureKeyCredential(AI_KEY)
    )


    result = client.analyze(
        image_data=image_data,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.PEOPLE
        ],
    )
    return result




def annotate_image(image, items, is_person=False):
    draw = ImageDraw.Draw(image)
    for item in items:
        if is_person and item.confidence <= 0.2:
            continue
        r = item.bounding_box
        bbox = [(r.x, r.y), (r.x + r.width, r.y + r.height)]
        draw.rectangle(bbox, outline='cyan', width=3)
        if not is_person:
            draw.text((r.x, r.y), item.tags[0].name, fill='cyan')
    return image




def main():
    st.title("🧠 Azure Vision Image Analyzer")
    st.markdown("Analyze objects, people, tags, and captions from your image.")


    # Image input
    img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    img_cam = st.camera_input("Or take a photo")


    image_bytes = None
    image_source = None


    if img_file is not None:
        image_bytes = img_file.read()
        image_source = img_file
    elif img_cam is not None:
        image_bytes = img_cam.getvalue()
        image_source = img_cam


    if image_bytes:
        st.image(image_bytes, caption="Input Image", use_column_width=True)
        st.write("Analyzing...")


        try:
            result = analyze_image(image_bytes)


            if result.caption:
                st.subheader("📝 Caption")
                st.write(f"**{result.caption.text}** (confidence: {result.caption.confidence:.2%})")


            if result.tags:
                st.subheader("🏷️ Tags")
                for tag in result.tags.list:
                    st.write(f"- {tag.name} ({tag.confidence:.2%})")


            image = Image.open(io.BytesIO(image_bytes))


            if result.objects and result.objects.list:
                st.subheader("📦 Objects Detected")
                for obj in result.objects.list:
                    st.write(f"- {obj.tags[0].name} ({obj.tags[0].confidence:.2%})")
                annotated = annotate_image(image.copy(), result.objects.list)
                st.image(annotated, caption="Objects Annotated")


            if result.people and result.people.list:
                st.subheader("🧍 People Detected")
                for person in result.people.list:
                    if person.confidence > 0.2:
                        st.write(f"- Bounding Box: {person.bounding_box} (confidence: {person.confidence:.2%})")
                annotated_people = annotate_image(image.copy(), result.people.list, is_person=True)
                st.image(annotated_people, caption="People Annotated")


        except Exception as e:
            st.error(f"Error analyzing image: {e}")




if __name__ == '__main__':
    main()





