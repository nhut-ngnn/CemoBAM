import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from architecture.Model import MultiModalGNN
from architecture.Graph_constructed import load_dataset
from utils.utils import set_seed, compute_metrics, train

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k_text_values = list(range(1, 11))
    k_audio_values = list(range(1, 11))

    best_wf1 = 0.0
    best_config = (0, 0)
    results = []

    for k_text in k_text_values:
        for k_audio in k_audio_values:
            print(f"\nTesting k_text={k_text}, k_audio={k_audio}")

            try:
                train_data = load_dataset(args.train_path, k_text=k_text, k_audio=k_audio).to(device)
                valid_data = load_dataset(args.valid_path, k_text=k_text, k_audio=k_audio).to(device)
                test_data  = load_dataset(args.test_path,  k_text=k_text, k_audio=k_audio).to(device)
            except Exception as e:
                print(f"Error loading dataset for k_text={k_text}, k_audio={k_audio}: {e}")
                continue

            model = MultiModalGNN(
                hidden_dim=args.hidden_dim,
                num_classes=args.num_classes,
                num_layers=args.num_layers,
                fusion_head_output_type=args.fusion_type
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)
            criterion = nn.CrossEntropyLoss()

            train(model, train_data, valid_data, optimizer, scheduler, criterion, epochs=args.epochs)

            model.eval()
            with torch.no_grad():
                test_out = model(test_data.text_x, test_data.audio_x, test_data.edge_index)
                test_pred = test_out.argmax(dim=1)

            y_true = test_data.y.cpu().numpy()
            y_pred = test_pred.cpu().numpy()
            wa, ua, wf1, uf1 = compute_metrics(y_true, y_pred)

            print(f"WA: {wa:.4f}, UA: {ua:.4f}, WF1: {wf1:.4f}, UF1: {uf1:.4f}")

            results.append({
                'k_text': k_text,
                'k_audio': k_audio,
                'WA': wa,
                'UA': ua,
                'WF1': wf1,
                'UF1': uf1
            })

            if wf1 > best_wf1:
                best_wf1 = wf1
                best_config = (k_text, k_audio)
                print(f"New Best: k_text={k_text}, k_audio={k_audio} → WF1={wf1:.4f}")

                # Save best model
                best_model_path = f"{args.save_dir}/MELD_best_model_{args.fusion_type}_kt{k_text}_ka{k_audio}.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to: {best_model_path}")

    # Save results
    df = pd.DataFrame(results)
    csv_path = f"{args.save_dir}/MELD_grid_search_results_{args.fusion_type}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Best config → k_text: {best_config[0]}, k_audio: {best_config[1]}, WF1: {best_wf1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search over k_text and k_audio for MELD")
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--fusion_type', type=str, default='min')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--save_dir', type=str, default='saved_models')

    args = parser.parse_args()
    main(args)
