import streamlit as st, json
import torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt

st.title("Deteksi Kesegaran Ikan 🐟")
# Load mapping
with open("class_mapping.json") as f:
    class_names = json.load(f)

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("fish_freshness_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

uploaded = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])
cam = st.camera_input("atau pakai kamera")

img = None
if uploaded: img = Image.open(uploaded)
elif cam: img = Image.open(cam)

if img:
    st.image(img, caption="Input", use_column_width=True)
    x = transform(img).unsqueeze(0)

    preds = model(x)
    probs = F.softmax(preds, dim=1)[0].detach().numpy()
    top_idx = probs.argmax()
    st.success(f"Prediksi: **{class_names[str(top_idx)]}** ({probs[top_idx]*100:.1f}%)")

    # visualisasi top5
    top5 = probs.argsort()[-5:][::-1]
    labels = [class_names[str(i)] for i in top5]
    vals = probs[top5]

    fig, ax = plt.subplots()
    ax.barh(labels, vals)
    ax.invert_yaxis()
    ax.set_xlabel("Probabilitas")
    st.pyplot(fig)
