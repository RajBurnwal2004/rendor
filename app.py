from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Define the same ordinal encoding mappings used during training
ordinal_mappings = {
    "Movement": {"Solar": 0, "Smartwatch": 1, "Quartz": 2, "Automatic": 3, "Manual winding": 4},
    "Case material": {
        "Plastic": 0,
        "Gold-plated": 1,
        "Brass": 2,
        "Tungsten": 3,
        "Gold/Steel": 4,
        "Steel": 5,
        "Bronze": 6,
        "Aluminum": 7,
        "Sapphire crystal": 8,
        "Silver": 9,
        "Yellow gold": 10,
        "Ceramic": 11,
        "Titanium": 12,
        "Palladium": 13,
        "Red gold": 14,
        "White gold": 15,
        "Rose gold": 16,
        "Tantalum": 17,
        "Carbon": 18,
        "Platinum": 19
    },
    "Bracelet material": {
        "Brass": 0,
        "Gold-plated": 1,
        "Silver": 2,
        "Steel": 3,
        "Silicon": 4,
        "Gold/Steel": 5,
        "Ostrich skin": 6,
        "Calf skin": 7,
        "Lizard skin": 8,
        "Aluminium": 9,
        "Leather": 10,
        "Snake skin": 11,
        "Titanium": 12,
        "Ceramic": 13,
        "Textile": 14,
        "Satin": 15,
        "Shark skin": 16,
        "Red gold": 17,
        "Plastic": 18,
        "Rubber": 19,
        "Yellow gold": 20,
        "Crocodile skin": 21,
        "Alligator skin": 22,
        "White gold": 23,
        "Rose gold": 24,
        "Platinum": 25
    },
    "Condition": {
        "Used (Incomplete)": 0,
        "Used (Poor)": 1,
        "Used (Fair)": 2,
        "Used (Good)": 3,
        "New": 4,
        "Like new & unworn": 5,
        "Used (Very good)": 6
    },
    "Scope of delivery": {
        "No original box, no original papers": 0,
        "Original box, no original papers": 1,
        "Original papers, no original box": 2,
        "Original box, original papers": 3
    },
    "Gender": {
        "Women's watch": 0,
        "Men's watch/Unisex": 1
    },
    "Availability": {
        "Item is in stock": 0,
        "Item available on request": 1,
        "Item needs to be procured": 2
    },
    "Crystal": {
        "Mineral Glass": 0,
        "Plastic": 1,
        "Glass": 2,
        "Plexiglass": 3,
        "Sapphire crystal": 4
    },
    "Clasp": {
        "Fold clasp, hidden": 0,
        "Buckle": 1,
        "Jewelry clasp": 2,
        "Fold clasp": 3,
        "Double-fold clasp": 4,
        "No clasp": 5
    }
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs
        input_data = {
            "Movement": request.form["movement"],
            "Case material": request.form["case_material"],
            "Bracelet material": request.form["bracelet_material"],
            "Condition": request.form["condition"],
            "Scope of delivery": request.form["scope_of_delivery"],
            "Gender": request.form["gender"],
            "Availability": request.form["availability"],
            "Crystal": request.form["crystal"],
            "Clasp": request.form["clasp"]
        }
        
        # Convert categorical inputs into numerical using the stored encoding
        encoded_input = [
            ordinal_mappings[feature].get(input_data[feature], -1)  # Default to -1 if value is not found
            for feature in input_data
        ]

        # Convert to numpy array and reshape for prediction
        encoded_input = np.array(encoded_input).reshape(1, -1)

        # Predict price
        prediction = model.predict(encoded_input)

        return render_template("index.html", prediction=f"Predicted Watch Price: ${prediction[0]:.2f}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
