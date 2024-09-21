import ml_collections

def Config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.patch_size = 16
    config.decoder_out_channels = 16
    config.n_classes = 2
    config.n_layer = 4
    config.channel = 1024

    return config


