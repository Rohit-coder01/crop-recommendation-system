🌾 Advanced Crop Recommendation System
Welcome to the Advanced Crop Recommendation System — a Streamlit-powered machine learning app that suggests the best crop to plant based on soil and environmental conditions.

🚀 Built with:
Python
Streamlit

Scikit-learn (Random Forest Classifier)
📦 Features
✅ Predicts the most suitable crop based on:
Nitrogen (N)
Phosphorus (P)
Potassium (K)
Temperature (°C)
Humidity (%)
pH
Rainfall (mm)


✅ Displays model accuracy.
✅ Provides valid input ranges for user guidance.
✅ User-friendly web interface using Streamlit.

🖥️ Demo
Run the app locally:

bash
Copy
Edit
streamlit run app.py
Replace app.py with your Python file name if needed.

🛠️ How It Works
Uses a Random Forest Classifier trained on the Crop_recommendation.csv dataset.

Scales input features with StandardScaler.

Checks that user inputs fall within valid, expected ranges.

Outputs the top recommended crop.

📊 Valid Input Ranges
Parameter	Range
Nitrogen (N)	0–200
Phosphorus (P)	0–200
Potassium (K)	0–200
Temperature	8–50 °C
Humidity	10–100 %
pH	3–9
Rainfall	20–300 mm

These are shown under each input box for easy reference.

📂 Files
app.py → Main Streamlit application code.

Crop_recommendation.csv → Dataset for training the model.

🏗️ Setup Instructions
1️⃣ Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/crop-recommendation-app.git
cd crop-recommendation-app
2️⃣ Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the app:

bash
Copy
Edit
streamlit run app.py
🤝 Contributing
Pull requests are welcome!
Feel free to fork, open issues, or suggest improvements.

🌟 Acknowledgments
Thanks to the creators of the original Crop_recommendation.csv dataset.
