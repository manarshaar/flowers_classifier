import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import argparse
import json
import os
from typing import Dict
image_size = 224

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32)/255.0
    return image.numpy()

def predict(image_path, model, top_k=2):
    image = Image.open(image_path)
    processed_image = process_image(np.asarray(image))
    predictions = model.predict(np.expand_dims(processed_image, axis=0))
    top_values, top_indices = tf.math.top_k(predictions, top_k)

    top_probabilities = top_values.numpy().squeeze()
    top_indices = top_indices.numpy().squeeze()
    top_classes = [str(i) for i in top_indices]

    return top_probabilities, top_classes

def load_classes_names(path: str) -> Dict[str, str]:
    with open(path, "r") as f:
        return json.load(f)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Predict flower name")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("model_path", help="Path to saved model (.5)")
    parser.add_argument("--top_k", type=int, default=3, help="top K classes")
    parser.add_argument("--category_names", default='label_map.json', help="JSON mapping file to map class ids to class names (e.g. label_map.json)")
    return parser

def main():
    args = build_parser().parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError("Image not found")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError("Model not found")

    model = tf.keras.models.load_model( args.model_path, custom_objects={"KerasLayer": hub.KerasLayer} )
    
    probabilities, classes = predict(args.image_path, model, args.top_k)

    if args.category_names:
        class_names = load_classes_names(args.category_names)
        labels = [class_names[c] for c in classes]
    else:
        labels = classes

    print("\nTop prediction:")
    print(f"  class: {labels[0]}")
    print(f"  probability: {probabilities[0]:.6f}")

    print(f"\nTop {args.top_k} predictions:")
    for label, p in zip(labels, probabilities):
        print(f"  {label}: {p:.6f}")
    print("\n")
if __name__ == "__main__":
    main()