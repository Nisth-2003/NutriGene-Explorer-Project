from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('impact_on_gene_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)

# Map for displaying impact
impact_mapping = {0: "No Effect", 1: "Upregulation", -1: "Downregulation"}


def predict_impact(gene_name, nutrient):
    """Function to predict impact based on user inputs."""
    try:
        # Encode the user inputs
        gene_name_encoded = label_encoders['Gene_Name'].transform([gene_name])[0]
        nutrient_encoded = label_encoders['Nutrient'].transform([nutrient])[0]

        # Predict impact
        prediction_encoded = model.predict([[gene_name_encoded, nutrient_encoded]])[0]

        # Decode the prediction
        prediction = label_encoders['Impact_on_Gene'].inverse_transform([prediction_encoded])[0]
        return prediction
    except Exception as e:
        return str(e)


@app.route('/')
def home():
    """Route for the home page."""
    return render_template('home.html')


@app.route('/health', methods=['GET', 'POST'])
def health():
    """Route for the health prediction page."""
    if request.method == 'POST':
        # Retrieve user inputs from the form
        gene_name = request.form.get('gene_name')
        nutrient = request.form.get('nutrient')

        # Validate inputs
        if not gene_name or not nutrient:
            return render_template('health.html', error="Please provide both Gene Name and Nutrient.")

        try:
            # Predict the impact
            prediction = predict_impact(gene_name, nutrient)
            return render_template('health.html', prediction=prediction)
        except Exception as e:
            return render_template('health.html', error=f"An error occurred: {str(e)}")
    return render_template('health.html')


if __name__ == '__main__':
    app.run(debug=True)
