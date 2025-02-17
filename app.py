from flask import Flask, request, jsonify
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

app = Flask(__name__)

# Load trained model
model = joblib.load('cyp1a2SVM.pkl')

def preprocess_smiles(smiles, radius=2, n_bits=1024):  # Adjust n_bits to 1024
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Handle invalid SMILES
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fingerprint)  # Convert to NumPy array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get input data
    smiles = data.get('smiles')

    # Log the received SMILES string
    print("Received SMILES:", smiles)

    # Convert SMILES to features
    features = preprocess_smiles(smiles)

    if features is None:
        return jsonify({'error': 'Invalid SMILES'}), 400  # Handle invalid SMILES

    print("Feature vector shape:", len(features))  # Print input shape before prediction
    
    prediction = model.predict([features])  # Make prediction
    return jsonify({'prediction': int(prediction[0])})  # Return result

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
