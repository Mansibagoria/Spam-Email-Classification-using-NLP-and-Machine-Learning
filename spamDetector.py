import streamlit as st
import pickle

# Load the trained model and vectorizer
model_path = r'P3-Spam-Email-Classification-using-NLP-and-Machine-Learning-main\spam.pkl'
vectorizer_path = r'P3-Spam-Email-Classification-using-NLP-and-Machine-Learning-main\vectorizer.pkl'

# Ensure these files exist at the specified paths
model = pickle.load(open(model_path, 'rb'))
cv = pickle.load(open(vectorizer_path, 'rb'))


def main():
    # Title of the web application
    st.title("Email Spam Classification Application")
    
    # Description of the app
    st.write("This is a Machine Learning application to classify emails as spam or ham (not spam).")
    
    # Input area for the user to enter the email text
    st.subheader("Enter an Email to Classify")
    user_input = st.text_area("Enter an email to classify", height=150)
    
    # When the classify button is clicked
    if st.button("Classify"):
        if user_input:
            # Prepare the input for prediction
            data = [user_input]
            
            # Transform the input email into a vector using the trained vectorizer
            vec = cv.transform(data).toarray()
            
            # Make the prediction using the loaded model
            result = model.predict(vec)
            
            # Display the result based on the prediction
            if result[0] == 0:
                st.success("This is NOT a Spam Email")
            else:
                st.error("This is a Spam Email")
        else:
            st.write("Please enter an email to classify.")

# Run the Streamlit app
if __name__ == '__main__':
    main()
