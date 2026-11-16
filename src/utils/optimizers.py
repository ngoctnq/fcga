def get_optimizer(cfg, model):
    if cfg.train.optimizer == "Adam":
        from torch.optim import Adam
        return Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == "SGD":
        from torch.optim import SGD
        return SGD(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {cfg.train.optimizer}")
