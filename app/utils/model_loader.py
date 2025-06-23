import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

from .gradcam_utils import apply_gradcam_and_interpret
from .report_generator_live import build_patch_pdf_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Assuming 2-class output
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_and_visualize(image: Image.Image):
    """Runs prediction and Grad-CAM, returns visual outputs and interpretation."""
    image_tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(image_tensor)
    confidence = torch.softmax(outputs, dim=1)[0].tolist()

    # Grad-CAM and interpretation
    overlay_img, heatmap_img, clinical_note, focus_score = apply_gradcam_and_interpret(
        model=model,
        image=image,
        target_layer=model.layer4[-1],  # last conv layer
        device=device,
    )

    return {
        "overlay_image": overlay_img,
        "heatmap_image": heatmap_img,
        "original_image": image,
        "confidence": confidence,
        "clinical_note": clinical_note,
        "focus_score": focus_score,
    }
