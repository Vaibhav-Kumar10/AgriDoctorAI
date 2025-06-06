# 🌿 AgriDoctorAI

AgriDoctorAI is an AI-powered plant disease detection app built with **Streamlit** and **TensorFlow**. Users can upload an image of a plant leaf, and the app predicts the disease and its confidence level.

---

## 📸 Features

- Upload leaf images directly from your device.
- AI-based disease prediction using a pre-trained deep learning model.
- Real-time prediction with confidence score.
- Clean and modern UI using Streamlit.

---

## 🧠 Model Info

- Trained on a custom tomato leaf disease dataset.
- TensorFlow/Keras `.h5` model.
- Input shape: `(224, 224, 3)`.

---

## 🚀 Deployment on Render

### 🌐 1. Prerequisites

- [GitHub](https://github.com) account
- [Render](https://render.com) account

### 📁 2. Project Structure

```
AgriDoctorAI/
├── app.py
├── model/
│ └── plant_disease_model.h5
├── requirements.txt
├── setup.sh
```

### 🛠 3. Push to GitHub

1. Create a new GitHub repository.
2. Push your project files to it.

```bash
git init
git remote add origin https://github.com/your-username/AgriDoctorAI.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

### ☁️ 4. Deploy on Render

1. Visit [https://render.com](https://render.com)
2. Click **New Web Service**
3. Connect your GitHub repo
4. Fill in:

   - **Build Command**: `./setup.sh`
   - **Start Command**: `streamlit run app.py`
   - **Environment**: Python 3.10

5. Click **Create Web Service**

Render will set up your app and give you a live URL like:

```
https://agridoc-ai.onrender.com
```

---

## 🧪 Usage

1. Open the deployed app link.
2. Upload a leaf image (`.jpg`, `.png`, `.jpeg`).
3. Wait for the model to predict.
4. See the disease name and confidence score.

---

## 🧾 Requirements

```txt
streamlit
tensorflow==2.13.0
keras
pillow
numpy
opencv-python
```

---

## 🤝 Contributions

Contributions are welcome! Fork the repo and raise a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.
