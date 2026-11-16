import src.models

def get_model(cfg):
    model_cls = getattr(src.models, cfg.model.name)
    
    return model_cls(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=cfg.model.output_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        **cfg.model.params.dict(),
    ).to(cfg.device)
