# Compile the library
cd ../ccn_lib/
sh compile.sh
cd ../torch_geometric

# Number of epochs
epochs=1024

# Learning rate
learning_rate=1e-3

# Architecture
initial_hidden=16
message_sizes=16,16
message_mlp_sizes=16

# Activation
activation=relu
# activation=relu

# Multi-threading
nThreads=10

# Batch size
batch_size=4

# Dataset's directory
data_dir=data/ZINC

# Training program
python3 train_zinc_subset.py --data_dir=$data_dir --epochs=$epochs --learning_rate=$learning_rate --initial_hidden=$initial_hidden --message_sizes=$message_sizes --message_mlp_sizes=$message_mlp_sizes --nThreads=$nThreads --activation=$activation
