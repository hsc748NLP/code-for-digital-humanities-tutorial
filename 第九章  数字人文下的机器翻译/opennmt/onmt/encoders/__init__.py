"""Module defining encoders."""
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.ggnn_encoder import GGNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.transformer import TransformerEncoder

str2enc = {"ggnn": GGNNEncoder, "rnn": RNNEncoder, "brnn": RNNEncoder,
           "cnn": CNNEncoder, "transformer": TransformerEncoder,
           "mean": MeanEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc"]
