import streamlit as st
from PIL import Image
import base64
import io
import re
import requests
from bs4 import BeautifulSoup
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np,json

# Add a title to the app using st.title
#st.title(":violet[Bird Image Classifier] :bird:")
#st.title('A title with _italics_ :blue [colors] and ')
st.title(':orange[__Birdy__]: A Fun and Educational Web App for Children:bird:')

# Add some instructions or explanations using st.write
st.write(":violet[__This app uses a machine learning model to predict the name of the bird in the image and Generate description about the bird.__]")


with open("C:/Users/Umang/Downloads/class_names_name.json", "r") as f:
    class_names = json.load(f)

# machine learning part

# Load the model
model = load_model("C:/Users/Umang/Downloads/fine_tune_resnet.h5")
    
    
def DES(pred_class):
    result_url =  "https://en.wikipedia.org/wiki/{}".format(pred_class)
    response = requests.get(result_url)
    html = response.text
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Find the main content div that contains the paragraphs
    content_div = soup.find("div", class_="mw-parser-output")
    
    # Extract the text from the paragraphs excluding image captions
    text = ""
    for element in content_div:
        if element.name == "p":
            # Exclude paragraphs that contain image captions
            if not element.find("div", class_="thumbcaption"):
                paragraph_text = element.get_text()
                
                # Remove numeric references within square brackets using regex
                cleaned_text = re.sub(r'\[\d+\]', '', paragraph_text)
                
                text += cleaned_text
    
    # Split the text at each occurrence of a period followed by a space
    sentences = text.split(".")
    
    ans=""
    # Print each sentence on a new line
    for sentence in sentences:
        sentence=sentence.strip()
        if(sentence!=""): 
            ans+=sentence+". "
    return ans

# Define the path of the new background image file
new_background_image_path = "C://Users//Umang//Downloads//e1a89f6d08f4d1c57f9a850b87d01a7f.jpeg"

# Convert the new image file into a base64 encoded string
def get_base64_of_image(image_path):
    img = Image.open(image_path)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG") 
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Get the base64 string of the new image
new_image_base64 = get_base64_of_image(new_background_image_path)

# Set the new background image using CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url('data:image/png;base64,{new_image_base64}');
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# Move the file uploader and the text box to the sidebar using st.sidebar
uploaded_file = st.sidebar.file_uploader("Please Select the Image", type=["jpg", "png"])


# Assume you have uploaded an image file using st.file_uploader
if uploaded_file is not None:
    # Show a spinner while processing the image using st.spinner
    with st.spinner("Processing image..."):
        image = Image.open(uploaded_file)
        new_img=image
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        pred_class = class_names[str(np.argmax(pred))]
        # Print the predicted class
        #print('The image is:', pred_class)
        des=DES(pred_class)
        if (len(des) == 0):
            new_ans=[]
            for i in pred_class.split():
                new_ans.append(i)
            anss=""
            if(len(new_ans)>1):
                anss="_".join(new_ans)
            else: anss=new_ans[0]
            #st.sidebar.text_area("I'm sorry,I did not able to get the data, please tap on the link!")
            google_search_link = f'https://www.google.com/search?q={anss}'
            link_with_text = f'[**Google Search for {anss}**]({google_search_link})'
            st.sidebar.text_area(f"I'm sorry, I did not able to get the data, please tap on the link! [{link_with_text.format(pred_class=anss)}] for about the bird.", height=250)

        else: description = st.sidebar.text_area(des)
    st.header(f":orange[__{pred_class}__]")
    # Display the image in the background using st.image
    st.image(new_img, caption="Input image", use_column_width=True)
    
    # Display the bird name as large text using st.markdown with new syntax
    
    # Show a success message when done using st.success
    st.success("The image is classified successfully.")
