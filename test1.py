import torch
from iTransformer import iTransformer

# using solar energy settings

model = iTransformer(
    num_variates = 137,
    lookback_len = 96,                  # or the lookback length in the paper
    dim = 256,                          # model dimensions
    depth = 6,                          # depth
    heads = 8,                          # attention heads
    dim_head = 64,                      # head dimension
    pred_length = (12, 24, 36, 48),     # can be one prediction, or many
    num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
    use_reversible_instance_norm = True # use reversible instance normalization, proposed here https://openreview.net/forum?id=cGDAkQo1C0p . may be redundant given the layernorms within iTransformer (and whatever else attention learns emergently on the first layer, prior to the first layernorm). if i come across some time, i'll gather up all the statistics across variates, project them, and condition the transformer a bit further. that makes more sense
)

time_series = torch.randn(2, 96, 137)  # (batch, lookback len, variates)

preds = model(time_series)

# preds -> Dict[int, Tensor[batch, pred_length, variate]]
#       -> (12: (2, 12, 137), 24: (2, 24, 137), 36: (2, 36, 137), 48: (2, 48, 137))