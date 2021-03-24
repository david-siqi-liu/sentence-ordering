args = {
    'dir': '/content/drive/MyDrive/Colab Notebooks/sentence-ordering/',
    'labeled_filename': 'data/train.pkl',
    'pred_filename': 'data/test.pkl',
    'model_output_dir': 'trained/',
    'seed': 647,
    'num_train_docs': int(1e5),
    'num_val_docs': int(5e2),
    'model_type': 'bert-base-uncased',
    'do_lower_case': True,
    'graph_method': 'max_flow',
    'model_fsent': {
        'max_length': 64,
        'num_train_samples': int(1e4),
        'target_ratio': 0.5,
        'num_epochs': 5,
        'batch_size': 32,
        'lr': 5e-5,
        'adam_eps': 1e-8,
        'warmup_steps': 0
    },
    'model_pair': {
        'max_length': 128,
        'num_train_samples': int(3e4),
        'num_epochs': 30,
        'batch_size': 32,
        'lr': 5e-5,
        'adam_eps': 1e-8,
        'warmup_steps': 0
    }
}
