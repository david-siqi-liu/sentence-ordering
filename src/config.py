args = {
    'data_dir': 'data/',
    'labeled_filename': 'train.pkl',
    'pred_filename': 'test.pkl',
    'seed': 647,
    'num_train_docs': int(1e5),
    'num_val_docs': int(5e2),
    'model_type': 'bert-base-uncased',
    'do_lower_case': True,
    'graph_method': 'max_flow',
    'model_fsent': {
        'max_length': 128,
        'num_train_samples': int(1e4),
        'target_ratio': 0.5,
        'num_epochs': 3,
        'batch_size': 32,
        'lr': 5e-5,
        'adam_eps': 1e-8,
        'warmup_steps': 0
    },
    'model_pair': {
        'max_length': 256,
        'num_train_samples': int(5e4),
        'target_ratio': 0.5,
        'num_epochs': 10,
        'batch_size': 16,
        'lr': 5e-5,
        'adam_eps': 1e-8,
        'warmup_steps': 0
    }
}
