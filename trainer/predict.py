import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import numpy as np
from architecture.Model import MultiModalGNN
from architecture.Graph_constructed import load_dataset
from utils.utils import set_seed, compute_metrics

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.num_classes == 4:
        emotion_labels = ['Anger', 'Happiness', 'Neutral', 'Sadness']
    elif args.num_classes == 5:
        emotion_labels = ['Anger', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    else:
        emotion_labels = [f'Class {i}' for i in range(args.num_classes)]

    test_data = load_dataset(args.test_path, k_text=args.k_text, k_audio=args.k_audio, device=device)
    test_data = test_data.to(device)

    model = MultiModalGNN(
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        fusion_head_output_type=args.fusion_type
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(test_data.text_x, test_data.audio_x, test_data.edge_index)
        predictions = logits.argmax(dim=1)

        if hasattr(model, 'get_embeddings'):
            embeddings = model.get_embeddings(
                test_data.text_x, test_data.audio_x, test_data.edge_index
            ).cpu().numpy()
        else:
            print("[Warning] get_embeddings not implemented in model. Using logits for t-SNE.")
            embeddings = logits.cpu().numpy()

    y_true = test_data.y.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    wa, ua, wf1, uf1 = compute_metrics(y_true, y_pred)
    print(f"Test WA (Accuracy): {wa:.4f}, Test UA (Balanced Accuracy): {ua:.4f}, "
          f"Test WF1: {wf1:.4f}, Test UF1: {uf1:.4f}")

    print("Running t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(embeddings)

    df_tsne = pd.DataFrame({
        "tsne-1": tsne_result[:, 0],
        "tsne-2": tsne_result[:, 1],
        "label_idx": y_true,
        "label_name": [emotion_labels[i] for i in y_true]
    })

    plt.figure(figsize=(10, 8))
    sns.set_context("notebook", font_scale=1.4)
    palette = sns.color_palette("bright", len(emotion_labels))

    sns.scatterplot(
        x="tsne-1", y="tsne-2",
        hue="label_name",
        palette=palette,
        data=df_tsne,
        s=60,
        alpha=0.7,
        edgecolor="k"
    )

    #title_text = f"t-SNE of MultiModalGNN with {args.fusion_type.capitalize()} Fusion on {os.path.basename(args.test_path).split('_')[0]} Dataset"
    #plt.title(title_text, fontsize=18, weight='bold')
    plt.xlabel("")
    plt.ylabel("")
    #plt.xticks([])
    #plt.yticks([])
    plt.legend(title="Emotion", fontsize=13, title_fontsize=14, loc="best")
    plt.tight_layout()
    dataset_name = os.path.basename(args.test_path).split('_')[0]
    save_name = f"tsne_{dataset_name.lower()}_{args.fusion_type.lower()}.png"
    plt.savefig(save_name, dpi=500)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MultiModalGNN on test set with t-SNE visualization")
    parser.add_argument('--test_path', type=str, required=True, help="Path to test .pkl file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to saved model .pt file")
    parser.add_argument('--fusion_type', type=str, default='mean', help="Fusion type (e.g., min, mean, max)")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension of the GNN")
    parser.add_argument('--num_classes', type=int, default=5, help="Number of emotion classes")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of GNN layers")
    parser.add_argument('--k_text', type=int, default=7, help="Number of text neighbors for graph")
    parser.add_argument('--k_audio', type=int, default=8, help="Number of audio neighbors for graph")
    args = parser.parse_args()
    main(args)
