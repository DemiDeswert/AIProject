parts = []
url_base = 'https://detect.roboflow.com/'
endpoint = '[baseballcap]'
access_token = '?access_token=[gifyELTiCBNUpgk9OCWI]'
format = '&format=json'
confidence = '&confidence=10'
stroke='&stroke=5'
parts.append(url_base)
parts.append(endpoint)
parts.append(access_token)
parts.append(format)
parts.append(confidence)
parts.append(stroke)
url = ''.join(parts)


import glob
import requests
import base64
from base64 import decodebytes
import io
from PIL import Image
import time
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

def main():
    st.title('Pretrained model demo')
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        image = Image.open(io.BytesIO(image_data))
    else:
        image= "",
    result = st.button('Run on image')
    if result:
        buffered = io.BytesIO()
        image = image.convert("RGB")
        image.save(buffered, quality=90, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        headers = {'accept': 'application/json'}
        start = time.time()
        r = requests.post(url, data=img_str, headers=headers)
        print('post took ' + str(time.time() - start))

        # print(r)
        print(r.json())
        preds = r.json()
        detections = preds['predictions']
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for box in detections:
            color = "#4892EA"
            x1 = box['x'] - box['width'] / 2
            x2 = box['x'] + box['width'] / 2
            y1 = box['y'] - box['height'] / 2
            y2 = box['y'] + box['height'] / 2

            draw.rectangle([
                x1, y1, x2, y2
            ], outline=color, width=5)

            if True:
                text = box['class']
                text_size = font.getsize(text)

                # set button size + 10px margins
                button_size = (text_size[0]+20, text_size[1]+20)
                button_img = Image.new('RGBA', button_size, color)
                # put text on button with 10px margins
                button_draw = ImageDraw.Draw(button_img)
                button_draw.text((10, 10), text, font=font, fill=(255,255,255,255))

                # put button on source image in position (0, 0)
                image.paste(button_img, (int(x1), int(y1)))
        imgpreditction=image.show()
        st.image(imgpreditction)

if __name__ == '__main__':
    main()




