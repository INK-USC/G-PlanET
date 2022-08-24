import argparse


class BaseParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument("--run_name", default="debug", type=str)
        self.add_argument("--num_train_epochs", default=5, type=int)
        self.add_argument("--evaluate_generated_text", default=True)
        self.add_argument("--evaluate_during_training", default=False)
        self.add_argument("--evaluate_during_training_verbose", default=False)
        self.add_argument("--save_model_every_epoch", default=False)
        self.add_argument("--train_batch_size", default=16, type=int)
        self.add_argument("--learning_rate", default=3e-5, type=float)
        self.add_argument("--max_length", default=256, type=int)
        self.add_argument("--max_seq_length", default=128, type=int)
        self.add_argument("--eval_batch_size", default=16, type=int)
        self.add_argument("--output", default='output', type=str)
        self.add_argument("--n_gpu", default=1, type=int)

class TrainParser(BaseParser):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument("--train_data_path", required=True, default="data")
        self.add_argument("--eval_data_path", required=True, default="data")
        self.add_argument("--model", type=str, default='facebook/bart-large')

class TestParser(BaseParser):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument("--test_data_path", required=True, default="data")
        self.add_argument("--model_path", required=True, default="data")
        self.add_argument("--num_beams", type=int, default=4)


