# coding=utf-8
 
"""
Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint.
"""
 
import numpy as np
import tensorflow as tf
import torch
from transformers import BertModel
import os
 
def convert_pytorch_checkpoint_to_tf(model: BertModel, ckpt_dir: str, model_name: str):
 
    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:
    Currently supported Huggingface models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """
 
    tensors_to_transpose = ("dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")
 
    var_map = (
        ("layer.", "layer_"),
        ("word_embeddings.weight", "word_embeddings"),
        ("position_embeddings.weight", "position_embeddings"),
        ("token_type_embeddings.weight", "token_type_embeddings"),
        (".", "/"),
        ("LayerNorm/weight", "LayerNorm/gamma"),
        ("LayerNorm/bias", "LayerNorm/beta"),
        ("weight", "kernel"),
    )
 
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
 
    state_dict = model.state_dict()
 
    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return "bert/{}".format(name)
 
    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var
 
    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))
 
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_").replace(".ckpt", "") + ".ckpt"))
 
def convert(pytorch_bin_path: str, pytorch_bin_model: str, tf_ckpt_path: str, tf_ckpt_model: str):
 
    model = BertModel.from_pretrained(
        pretrained_model_name_or_path=pytorch_bin_path,
        state_dict=torch.load(os.path.join(pytorch_bin_path, pytorch_bin_model), map_location='cpu')
    )
 
    convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir=tf_ckpt_path, model_name=tf_ckpt_model)
 
if __name__ == '__main__':
    bin_path = r'/home/admin/pretrain_models/sikuroberta_vocabtxt'
    bin_model = 'pytorch_model.bin'
    ckpt_path = r'/home/admin/pretrain_models/sikuroberta_vocabtxt_ckpt'
    ckpt_model = 'bert_model.ckpt'
 
    convert(bin_path, bin_model, ckpt_path, ckpt_model)

