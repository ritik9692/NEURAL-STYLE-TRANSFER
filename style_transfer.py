import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')

    # Resize
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)

# Load content and style images
content = load_image("content.jpg")
style = load_image("style.jpg", shape=[content.size(2), content.size(3)])

# Display function
def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
    image = image.clip(0, 1)
    return image

# Feature extractor from VGG19
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Freeze parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Layers for content and style
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Name each layer for easier access
def get_features(image, model, layers=None):
    features = {}
    x = image
    i = 0
    for layer in model.children():
        x = layer(x)
        name = None
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv{i}_1"
        elif isinstance(layer, nn.ReLU):
            name = f"relu{i}_1"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool{i}_1"
        if name in layers:
            features[name] = x
    return features

# Gram matrix for style
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Get features
content_features = get_features(content, vgg, content_layers + style_layers)
style_features = get_features(style, vgg, content_layers + style_layers)

# Compute gram matrices for style features
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# Target image (copy of content)
target = content.clone().requires_grad_(True).to(device)

# Weights for losses
style_weights = {'conv1_1': 1.0, 'conv2_1': 0.75, 'conv3_1': 0.2,
                 'conv4_1': 0.2, 'conv5_1': 0.2}
content_weight = 1e4
style_weight = 1e2

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Training loop
steps = 2000
for step in range(1, steps+1):
    target_features = get_features(target, vgg, content_layers + style_layers)

    # Content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    # Style loss
    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        _, d, h, w = target_feature.shape
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2) / (d * h * w)
        style_loss += layer_style_loss

    # Total loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    # Backprop
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Logging
    if step % 500 == 0:
        print(f"Step {step}, Total Loss: {total_loss.item()}")

# Show final result
plt.imshow(im_convert(target))
plt.title("Stylized Image")
plt.axis('off')
plt.show()