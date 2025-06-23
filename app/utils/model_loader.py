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
    pred = outputs.argmax(dim=1).item()

    # Grad-CAM and interpretation
    overlay_img, heatmap_img, note, focus_score, center_coords, observation =  = apply_gradcam_and_interpret(
        model=model,
        image=image,
        target_layer=model.layer4[-1],
        class_idx=pred,
        device=device
    )

    center_coords = (0.50, 0.50)  # Replace with real logic if available

        return {
        "pred": pred,
        "conf": confidence[pred],
        "overlay": overlay_img,
        "heatmap": heatmap_img,
        "original": image,
        "note": note,
        "observation": observation,
        "focus_percent": f"{focus_score * 100:.2f}%",
        "center": f"({center_coords[0]:.2f}, {center_coords[1]:.2f})",
        "pdf_buffer": build_patch_pdf_report(
            original_image=image,
            heatmap_image=heatmap_img,
            overlay_image=overlay_img,
            prediction=pred,
            confidence=confidence[pred],
            focus_ratio=focus_score,
            center_distance=center_coords,
            observation=observation,
            clinical_note=note
        )
    }
