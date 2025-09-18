from torch.utils.data import DataLoader
from datasets import build_dataset


def tuI(config):
    print("tuI bullied successfully   ")
    if config.data_mode == 'train':
        train_data , _ = build_dataset(is_train=True, args=config)
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))
        return train_dataloader

    if config.data_mode == 'test':
        test_data , _ = build_dataset(is_train=False, args=config)
        test_dataloader = DataLoader(test_data,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader)))
        return test_dataloader
