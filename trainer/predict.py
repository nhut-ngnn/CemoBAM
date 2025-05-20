import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from architecture.Model import MultiModalGNN
from architecture.Graph_constructed import load_dataset
from utils.utils import set_seed, compute_metrics

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        test_out = model(test_data.text_x, test_data.audio_x, test_data.edge_index)
        test_pred = test_out.argmax(dim=1)

    y_true = test_data.y.cpu().numpy()
    y_pred = test_pred.cpu().numpy()

    wa, ua, wf1, uf1 = compute_metrics(y_true, y_pred)
    print(f"Test WA (Accuracy): {wa:.4f}, Test UA (Balanced Accuracy): {ua:.4f}, "
          f"Test WF1: {wf1:.4f}, Test UF1: {uf1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MultiModalGNN on test set")
    parser.add_argument('--test_path', type=str, required=True, help="Path to test .pkl file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to saved model .pt file")
    parser.add_argument('--fusion_type', type=str, default='min', help="Fusion type (e.g., min, mean, max)")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension")
    parser.add_argument('--num_classes', type=int, default=4, help="Number of emotion classes")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of GNN layers")
    parser.add_argument('--k_text', type=int, default=7, help="Number of text neighbors")
    parser.add_argument('--k_audio', type=int, default=8, help="Number of audio neighbors")

    args = parser.parse_args()
    main(args)
