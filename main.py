import streamlit as st
import tensorflow as tf
import numpy as np



def model_prediction(test_image):
    model = tf.keras.layers.TFSMLayer("final_model", call_endpoint="serving_default")
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # batch
    
    predictions = model(input_arr)   # ✅ correct way
    
    # get actual output tensor from dict
    predictions = list(predictions.values())[0]
    
    return np.argmax(predictions)

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "image.jpg"
    st.image(image_path,width="stretch")
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))




# Sidebar Chatbot
st.sidebar.header("Plant Bot 🤖")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.sidebar.text_input("You:", key="user_input")
responses = {
        # Greetings
    "hello": "Hi there! How can I help you with plant diseases today? 🌿",
    "hi": "Hello! Feel free to ask me about plant diseases.",
    "how are you": "I'm just a bot, but I'm ready to help you detect plant diseases!",

    "how to use the app": """
    Here's how to use the Plant Disease Recognition System 🌿:

    1. **Home Page** – Learn about the app and what it does.
    2. **Disease Recognition Page** – Upload an image of your plant leaf.
    3. **Predict** – Click the **Predict** button to see which disease (if any) your plant has.
    4. **Results** – The system will show the predicted disease and its details.
    5. **About Page** – Learn about the dataset, project, and supported crops.

    You can also ask me about specific crops or ask for explanations of plant diseases!
    """,   # ✅ <-- COMMA ADDED HERE
    
    # App navigation
    "disease": "You can upload a plant image on the Disease Recognition page to detect diseases.",
    "help": "You can ask me things like 'how to detect disease', 'what crops are supported', or just say hi!",
    "about": "Check out the About page to learn about the dataset and project details.",

    # Crop-specific
    "tomato": "We can detect Tomato diseases like Early Blight, Leaf Mold, Septoria Leaf Spot, and more.",
    "apple": "Apple diseases include Apple Scab, Cedar Apple Rust, Black Rot, and more.",
    "corn": "Corn diseases include Common Rust, Northern Leaf Blight, and Gray Leaf Spot.",
    "potato": "Potato diseases include Early Blight and Late Blight.",
    "grape": "Grape diseases include Black Rot, Esca, and Leaf Blight.",
    "strawberry": "Strawberry diseases include Leaf Scorch and Powdery Mildew.",

    # Explanations
    "explain tomato": "Tomato diseases: \n- Early Blight: Dark spots on leaves, stems, or fruit.\n- Leaf Mold: Yellowing leaves with fuzzy growth.\n- Septoria Leaf Spot: Small dark spots with gray centers.",
    "explain apple": "Apple diseases: \n- Apple Scab: Dark olive spots on leaves and fruit.\n- Cedar Apple Rust: Orange/yellow spots on leaves.\n- Black Rot: Fruit rot with dark lesions.",
    "explain corn": "Corn diseases: \n- Common Rust: Reddish-brown pustules on leaves.\n- Northern Leaf Blight: Long gray-green lesions.\n- Gray Leaf Spot: Small rectangular lesions on leaves.",
    "explain potato": "Potato diseases: \n- Early Blight: Brown spots with concentric rings.\n- Late Blight: Dark, water-soaked lesions on leaves and tubers.",
    "explain grape": "Grape diseases: \n- Black Rot: Dark brown to black lesions on leaves and fruit.\n- Esca: Leaf discoloration and fruit rot.\n- Leaf Blight: Brown necrotic areas on leaves.",
    "explain strawberry": "Strawberry diseases: \n- Leaf Scorch: Red/brown lesions on leaf edges.\n- Powdery Mildew: White powdery coating on leaves.",

    # General tips
    "prevent disease": "Make sure to water plants at the base, remove infected leaves, and rotate crops regularly.",
    "organic": "Using neem oil or natural fungicides can help prevent plant diseases organically.",
    "fertilizer": "Balanced fertilizers help plants grow healthy and resist diseases.",
    "pests": "Regularly inspect plants for pests and use natural remedies or pesticides if necessary.",

    # Fun / small talk
    "thank you": "You're welcome! 🌱",
    "thanks": "Anytime! Take care of your plants! 🌿",
    "bye": "Goodbye! Happy gardening! 🌸",
    "good morning": "Good morning! Ready to check some plants today? ☀️",
    "good night": "Good night! Sweet dreams for you and your plants! 🌙"
}

if user_input:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "message": user_input})

    # Generate bot response
    response_found = False
    for key in responses:
        if key in user_input.lower():
            st.session_state.chat_history.append({"role": "bot", "message": responses[key]})
            response_found = True
            break
    if not response_found:
        st.session_state.chat_history.append({"role": "bot", "message": "Sorry, I didn't understand that. 🤔"})

# Display chat history newest on top
for chat in reversed(st.session_state.chat_history):
    if chat["role"] == "user":
        st.sidebar.markdown(f"**You:** {chat['message']}")
    else:
        st.sidebar.markdown(f"**Bot:** {chat['message']}")

