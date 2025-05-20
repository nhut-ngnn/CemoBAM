import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from architecture.Model import MultiModalGNN
from architecture.Graph_constructed import load_dataset
from utils.utils import set_seed, compute_metrics, train

def main(args):
    metrics = {'WA': [], 'UA': [], 'WF1': [], 'UF1': []}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_path, exist_ok=True)

    for seed in args.seeds:
        print(f"\n=== Running seed {seed} ===")
        set_seed(seed)

        train_data = load_dataset(args.train_path, k_text=args.k_text, k_audio=args.k_audio, device=device)
        valid_data = load_dataset(args.valid_path, k_text=args.k_text, k_audio=args.k_audio, device=device)
        test_data  = load_dataset(args.test_path,  k_text=args.k_text, k_audio=args.k_audio, device=device)

        model = MultiModalGNN(args.hidden_dim, args.num_classes, num_layers=args.num_layers, 
                              fusion_head_output_type=args.fusion_type).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=15)
        criterion = nn.CrossEntropyLoss()

        train(model, train_data, valid_data, optimizer, scheduler, criterion, epochs=args.epochs)

        model_path = os.path.join(args.save_path, f"{args.model_name}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        model.eval()
        test_data = test_data.to(device)
        with torch.no_grad():
            test_out = model(test_data.text_x, test_data.audio_x, test_data.edge_index)
            test_pred = test_out.argmax(dim=1)

        y_true = test_data.y.cpu().numpy()
        y_pred = test_pred.cpu().numpy()
        wa, ua, wf1, uf1 = compute_metrics(y_true, y_pred)

        print(f"Seed {seed} - WA: {wa:.4f}, UA: {ua:.4f}, WF1: {wf1:.4f}, UF1: {uf1:.4f}")
        metrics['WA'].append(wa)
        metrics['UA'].append(ua)
        metrics['WF1'].append(wf1)
        metrics['UF1'].append(uf1)

    print("\n=== Average Results ===")
    print(f"Avg WA:  {np.mean(metrics['WA']):.4f} ± {np.std(metrics['WA']):.4f}")
    print(f"Avg UA:  {np.mean(metrics['UA']):.4f} ± {np.std(metrics['UA']):.4f}")
    print(f"Avg WF1: {np.mean(metrics['WF1']):.4f} ± {np.std(metrics['WF1']):.4f}")
    print(f"Avg UF1: {np.mean(metrics['UF1']):.4f} ± {np.std(metrics['UF1']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='saved_models')
    parser.add_argument('--model_name', type=str, default='IEMOCAP_HemoGAT_min')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--fusion_type', type=str, default='min')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--k_text', type=int, default=7)
    parser.add_argument('--k_audio', type=int, default=8)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])

    args = parser.parse_args()
    main(args)
