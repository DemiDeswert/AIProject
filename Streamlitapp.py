
import streamlit as st
from roboflow import Roboflow

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
def load_model():
    rf = Roboflow(api_key="gifyELTiCBNUpgk9OCWI")
    project = rf.workspace().project("baseballcap")
    model = project.version(3).model
    return model
def main():
    st.title('Pretrained model demo')
    model = load_model()
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        model.predict(image, confidence=40, overlap=30).save("prediction.jpg")
        prediction = Image.open('prediction.jpg')
        st.image(prediction)

if __name__ == '__main__':
    main()




