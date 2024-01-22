# pip install streamlit, python-dotenv
import os
from clarifai.client.model import Model
from clarifai.client.input import Inputs
import streamlit as st

clarifai_pat = os.getenv('CLARIFAI_PAT')

def normalize_input(input_str):
    # Normalize input to lower case and strip spaces
    return [item.strip().lower() for item in input_str.split(',') if item.strip()]

def check_allergy(allergy, ingredients_list):
    # Additional check for common variations or synonyms of allergens
    allergy_variants = {
        'shrimp': ['shrimp', 'prawns'],
        # Add more allergens and their common variants here if needed
    }
    # Check for both the allergy and its variants
    variants = allergy_variants.get(allergy, [allergy])
    return any(variant in ingredients_list for variant in variants)

def get_user_health_info():
    health_info = {
        'allergies': [],
        'dietary_restrictions': [],
        'health_conditions': []
    }
    allergies = st.text_input("Do you have any food allergies? (Enter comma-separated list or 'None')")
    dietary_restrictions = st.text_input("Any dietary restrictions? (e.g., vegetarian, gluten-free, etc. or 'None')")
    health_conditions = st.text_input("Any health conditions to consider? (e.g., diabetes, hypertension, etc. or 'None')")

    if allergies.lower().strip() != 'none':
        health_info['allergies'] = normalize_input(allergies)
    if dietary_restrictions.lower().strip() != 'none':
        health_info['dietary_restrictions'] = normalize_input(dietary_restrictions)
    if health_conditions.lower().strip() != 'none':
        health_info['health_conditions'] = normalize_input(health_conditions)

    return health_info

def analyze_ingredients(ingredients, health_info):
    advice = ""
    ingredients_list = normalize_input(ingredients)

    # Check for allergies
    for allergy in health_info['allergies']:
        if check_allergy(allergy, ingredients_list):
            advice += f"Warning: This dish contains {allergy}, which you are allergic to.\n"

    # Rest of the checks remain the same...

    return advice if advice else "This food is suitable for you."

def main():
    st.title("ECO LIFE 360")
    file_uploaded = st.file_uploader("Upload an Image File", type=["jpg", "jpeg", "png"])

    with st.sidebar:
        st.text('Add your Clarifai PAT')
        clarifai_pat = st.text_input('Clarifai PAT:', type='password')
    
    if not clarifai_pat:
        st.warning('Please enter your PAT to continue!', icon='⚠️')
        return  # Stop execution if PAT is not provided
    
    if file_uploaded is not None:
        os.environ['CLARIFAI_PAT'] = clarifai_pat
        file_bytes = file_uploaded.read()
        st.image(file_uploaded, caption="Uploaded Image", use_column_width=True)

        prompt = "Identify the ingredients in this dish."
        inference_params = dict(temperature=0.4, max_tokens=100)

        model_prediction = Model("https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision").predict(inputs=[Inputs.get_multimodal_input(input_id="", image_bytes=file_bytes, raw_text=prompt)], inference_params=inference_params)
        ingredients = model_prediction.outputs[0].data.text.raw
        st.write("Identified Ingredients:", ingredients)

        user_health_info = get_user_health_info()

        # Only show advice after user has entered their health information
        if st.button('Analyze Ingredients'):
            advice = analyze_ingredients(ingredients, user_health_info)
            st.write(advice)
    else:
        st.warning('Please upload an image to proceed.', icon='⚠️')

if __name__ == "__main__":
    main()