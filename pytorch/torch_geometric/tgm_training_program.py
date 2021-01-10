# Training program for Covariant Compositional Networks (CCN) 1D

# from absl import flags
# from absl import app
import argparse
import time
import torch
import CCN1D
from TGMGraph import TGMGraph
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from tqdm import tqdm

dtype = torch.float
device = torch.device("cpu")


def build_parser():
    parser = argparse.ArgumentParser(description='Command description.')
    parser.add_argument('--data_dir', type=str, default='', help='Dataset directory')
    parser.add_argument('--data_name', type=str, default='', help='Data name')
    parser.add_argument('--epochs', type=int, default=1024, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--initial_hidden', type=int, default=16, help='Input size')
    parser.add_argument('--message_sizes', type=str, default='', help='Message sizes')
    parser.add_argument('--message_mlp_sizes', type=str, default='', help='Multi-layer perceptron sizes')
    parser.add_argument('--nThreads', type=int, default=14, help='Number of threads')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--activation', type=str, default='relu', help='Activation')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    return parser


def train_epoch(train_loader, model, optimizer):
    # Molecular graphs concatenation
    total_train_loss = 0
    for batch in tqdm(train_loader):
        graph = TGMGraph(batch, construct_adj=False)

        # Training
        optimizer.zero_grad()
        output = model(graph)
        loss = torch.nn.functional.l1_loss(output.squeeze(), graph.label)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * graph.num_graphs

    return total_train_loss / len(train_loader.dataset)


@torch.no_grad()
def val_epoch(val_loader, model):
    # Molecular graphs concatenation
    total_train_loss = 0
    for batch in val_loader:
        graph = TGMGraph(batch, construct_adj=False)
        output = model(graph)
        loss = torch.nn.functional.l1_loss(output.squeeze(), graph.label)
        total_train_loss += loss.item() * graph.num_graphs
    return total_train_loss / len(val_loader.dataset)


def setup_data_loaders(data_folder, batch_size=32, run_test=False, num_workers=3):
    train_dataset = ZINC(data_folder, subset=True, split='train')
    val_dataset = ZINC(data_folder, subset=True, split='val')
    if run_test:
        test_dataset = ZINC(data_folder, subset=True, split='test')

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    if run_test:
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


def main(args=None):
    parser = build_parser()
    args = parser.parse_args(args=args)

    data_dir = args.data_dir
    epochs = args.epochs
    learning_rate = args.learning_rate
    initial_hidden = args.initial_hidden
    message_sizes = [int(element) for element in args.message_sizes.strip().split(',')]
    message_mlp_sizes = [int(element) for element in args.message_mlp_sizes.strip().split(',')]
    nThreads = args.nThreads
    batch_size = args.batch_size
    activation = args.activation

    train_loader, val_loader, __ = setup_data_loaders(data_dir, batch_size)

    # Model creation
    initial_hidden = [initial_hidden]

    model = CCN1D.CCN1D(initial_hidden, message_sizes, message_mlp_sizes, 1, nThreads, activation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)

    print('\n--- Training -------------------------------')

    for epoch in range(epochs):
        print('\nEpoch ' + str(epoch))

        start_time = time.time()
        train_loss = train_epoch(train_loader, model, optimizer)
        # Training
        end_time = time.time()

        print('Training time = ' + str(end_time - start_time))
        print('Training loss = ' + str(train_loss))

        # Validation
        start_time = time.time()
        val_loss = train_epoch(train_loader, model, optimizer)
        end_time = time.time()

        print('Testing time = ' + str(end_time - start_time))
        print('Testing accuracy = ' + str(val_loss))


if __name__ == '__main__':
    main()
