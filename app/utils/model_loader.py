import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
from gradcam_utils import apply_gradcam_and_interpret
from utils.report_generator_live import build_patch_pdf_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/resnet18_epoch10_20250615_024251.pth", map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.7, 0.7, 0.7], [0.25, 0.25, 0.25])
])

def predict_and_visualize(image_file):
    image = Image.open(image_file).convert("RGB")
    tensor = transform(image).to(device)

    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        conf = torch.softmax(logits, dim=1).max().item()
        pred = torch.argmax(logits).item()

    original_image, heatmap, overlay, observation, clinical_note, center_distance, focus_ratio = apply_gradcam_and_interpret(
        model, tensor, pred, image
    )

    pdf_buffer = build_patch_pdf_report(
        original_image=original_image,
        heatmap_image=heatmap,
        overlay_image=overlay,
        prediction=pred,
        confidence=conf,
        focus_ratio=focus_ratio,
        center_distance=center_distance,  # This is a tuple (x, y)
        observation=observation,
        clinical_note=clinical_note
    )


    return {
        "original": original_image,
        "heatmap": heatmap,
        "overlay": overlay,
        "pred": pred,
        "conf": conf,
        "focus_percent": round(focus_ratio * 100, 2),
        "center": center_distance,
        "observation": observation,
        "note": clinical_note,
        "pdf_buffer": pdf_buffer
    }