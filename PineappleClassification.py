import typer
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from lime import lime_image
import sys
from PIL import Image

# Pineapple Classification Script
# This script processes images in a specified folder, classifies them using a pre-trained model,
# and optionally saves results in CSV format or applies explainable AI techniques.

img_height  = 224 # height of the image
img_width   = img_height

num_features=5

MODEL_PATH = 'Models/MULTILABEL_AE_2025-06-21_17-20-06.keras'

def main(folder: str, csv: bool = False, lime: bool = False):
    # Load the pre-trained model
    model_path = MODEL_PATH  # Update with your model path
    if not os.path.exists(model_path):
        print(f"\nModel file '{model_path}' does not exist.\n")
        return
    model = load_model(model_path)
    print(f"\nModel loaded from {model_path}.\n")

    if not os.path.exists(folder):
        print(f"\nFolder '{folder}' does not exist.\n")
        return
    if not os.path.isdir(folder):
        print(f"\n'{folder}' is not a directory.\n")
        return
    print(f"\nProcessing folder: {folder}\n")

    if lime:
        # 3. Set up LIME explainer
        explainer = lime_image.LimeImageExplainer()
        lime_output_folder = "lime_outputs"
        os.makedirs(lime_output_folder, exist_ok=True)

    if csv:
        results = []

    # List all files in the directory with jpeg, png, and csv extensions
    valid_extensions = ('.jpeg', '.jpg')
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
    if not files:
        print("\nNo valid image files found in the directory.\n")
        return
    print(f"\nFound {len(files)} valid files in the directory.\n")

    for file in files:
        print(f"\n")
        img_path = os.path.join(folder, file)
        img = image.load_img(img_path, 
                             target_size=(img_height, img_width), 
                             color_mode='rgb')
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        
        preds = model.predict(x, verbose=0)

        # second output neuron gives us the probability of browning
        print(f"{file} => probability of browning: {preds[0][1]:.2f}") 
        
        # first output neuron gives us the probability of translucency
        print(f"{file} => probability of translucency: {preds[0][0]:.2f}")
        
        if csv:
            results.append({
                'file': file,
                'browning_probability': round(float(preds[0][1]), 2),
                'translucency_probability': round(float(preds[0][0]), 2)
            })

        if lime:
            # 4. Explain prediction for the first label (e.g., has_translucency = 0)
            sys.stdout = open(os.devnull, 'w') # Suppress stdout
            explanation = explainer.explain_instance(
                x[0],                            # image in [0,1]
                model.predict,                   # function to get predictions
                top_labels=2,                    # number of labels to show
                hide_color=0,
                num_samples=1000,                # number of samples for LIME
            )
            sys.stdout = sys.__stdout__ # Restore stdout

            temp, _ = explanation.get_image_and_mask(
                label=0, # translucency
                positive_only=False,
                num_features=num_features,
                hide_rest=False
            )

            lime_filename = os.path.join(
                lime_output_folder, f"{os.path.splitext(file)[0]}_translucency_Prob_{int(100*preds[0][0])}.jpg"
            )

            # Make sure temp is uint8
            if temp.dtype != np.uint8:
                temp = (temp * 255).astype(np.uint8)
            img_to_save = Image.fromarray(temp)
            img_to_save.save(lime_filename, format="JPEG")
            print(f"LIME explanation saved to {lime_filename}")


            temp, _ = explanation.get_image_and_mask(
                label=1, # browning
                positive_only=False,
                num_features=num_features,
                hide_rest=False
            )

            lime_filename = os.path.join(
                lime_output_folder, f"{os.path.splitext(file)[0]}_browning_Prob_{int(100*preds[0][1])}.jpg"
            )

            # Make sure temp is uint8
            if temp.dtype != np.uint8:
                temp = (temp * 255).astype(np.uint8)
            img_to_save = Image.fromarray(temp)
            img_to_save.save(lime_filename, format="JPEG")
            print(f"LIME explanation saved to {lime_filename}")


    if csv:
        results_df = pd.DataFrame(results)
        results_df.to_csv('classification_results.csv', index=False)
        print(f"\nResults saved to classification_results.csv\n")


if __name__ == "__main__":
    # Command-line interface for the script
    # run using: python PineappleClassification.py <images folder> --csv --lime
    typer.run(main)
