import streamlit as st
import pickle
from PIL import Image
import io

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
def load_model():
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    return pickled_model 

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
