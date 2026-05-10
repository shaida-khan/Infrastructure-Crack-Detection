import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

CLASSES = ["Crack", "No Crack"]

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("crack_detection_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image):
    if image is None:
        return "Please upload an image first.", ""

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = CLASSES[predicted.item()]
    conf = confidence.item() * 100

    result = f"Prediction: {label}"
    details = f"Confidence: {conf:.2f}%"

    return result, details


custom_css = """
.gradio-container {
    background: #0f172a !important;
    color: white !important;
}

#header {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 20px;
    border: 1px solid #475569;
}

#header h1 {
    color: #facc15;
    font-size: 34px;
    margin-bottom: 8px;
}

#header p {
    color: #e2e8f0;
    font-size: 17px;
}

.card {
    background: #111827;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #334155;
}

.gr-button {
    background: #facc15 !important;
    color: #111827 !important;
    font-weight: bold !important;
    border-radius: 12px !important;
}
"""

with gr.Blocks(css=custom_css, title="Infrastructure Crack Detection") as demo:

    gr.HTML("""
    <div id="header">
        <h1>🧱 Infrastructure Crack Detection</h1>
        <p>AI-powered computer vision tool for detecting cracks in concrete and infrastructure images.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Upload Image")
            image_input = gr.Image(type="pil", label="Concrete / Infrastructure Image")
            analyze_btn = gr.Button("Analyze Image")

        with gr.Column():
            gr.Markdown("### 🔍 AI Prediction")
            result_output = gr.Textbox(label="Result")
            confidence_output = gr.Textbox(label="Confidence")

    analyze_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[result_output, confidence_output]
    )

    gr.Markdown("""
    ### 🏗️ Real-World Use Cases
    - Building and bridge inspection  
    - Road and pavement crack screening  
    - Concrete surface condition monitoring  
    - Civil engineering maintenance support  

    ### ⚙️ Tech Stack
    **Python · PyTorch · Torchvision · MobileNetV2 · Gradio · Hugging Face Spaces**
    """)

demo.launch(server_port=7861)