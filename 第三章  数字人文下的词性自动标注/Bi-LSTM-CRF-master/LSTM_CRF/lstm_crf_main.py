#encoding=utf-8
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode

from LSTM_CRF import lstm_crf_data_helper,lstm_crf_model
tf.flags.DEFINE_integer("embeddings_size", 300, "每个字向量的维度")
tf.flags.DEFINE_integer("hidden_dim", 300, "LSTM隐藏层细胞的个数")


tf.flags.DEFINE_integer("batch_size", 128, "每个批次的大小")
tf.flags.DEFINE_integer("num_epochs", 9, "训练的轮数")
tf.flags.DEFINE_float("keep_prob", 0.5, "丢失率")
tf.flags.DEFINE_float("clip_grad",5.0, "梯度的范围")
tf.flags.DEFINE_float("learning_rate",0.001, "学习率")
FLAGS = tf.flags.FLAGS
#用于每个字标记向索引的映射，注意“O”对应的必须是0因为标签的填充是以0进行填充的
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }
#第一维大小为batch_size第二维是句子的长度是动态获得的没法设置
x = tf.placeholder(tf.int32, [None, None], name='input')
y = tf.placeholder(tf.int32, [None, None], name='output')
sequence_lengths=tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
#获得字与字索引映射的向量
word2id_dict=lstm_crf_data_helper.get_word_id('data/word2id.pkl')
#获得总的标注类别数
num_tags=len(tag2label)
def get_data(file_location):
    #获得数据
    sentences_list,tags_list=lstm_crf_data_helper.get_data(file_location)
    #完成tag向索引的映射
    tags_id_list=lstm_crf_data_helper.tags2id(tags_list,tag2label)
    #对索引进行填充
    labels,_=lstm_crf_data_helper.padding_sentences(tags_id_list)
    #获得句子中每个字的id
    sentences_id_list=lstm_crf_data_helper.sentence2id(sentences_list,word2id_dict)
    #对句子或标注序列索引进行填充并获得每个句子的长度
    sen_index_list,sen_len_list=lstm_crf_data_helper.padding_sentences(sentences_id_list)
    return sen_index_list,labels,sen_len_list,tags_list
def backward_propagation():
    train_sen_index_list,train_labels,train_sen_len_list,_=get_data('data/train_data')
    test_sen_index_list,test_labels,test_sen_len_list,test_tags_list=get_data('data/test_data')
    # 首先是embedding层获得词向量数据
    with tf.name_scope("embedding"):
        embedding_mat=lstm_crf_data_helper.random_embedding(word2id_dict,FLAGS.embeddings_size)
        input_x=tf.nn.embedding_lookup(embedding_mat,x)
        # input_x=tf.nn.dropout(input_x,keep_prob=FLAGS.keep_prob)
    BiLSTM_CRF=lstm_crf_model.BiLSTM_CRF(FLAGS.hidden_dim,num_tags,input_x,sequence_lengths,keep_prob,y)
    loss,transition_params,logits=BiLSTM_CRF.positive_propagation()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    grads_and_vars = optim.compute_gradients(loss)
    grads_and_vars_clip = [[tf.clip_by_value(g,-FLAGS.clip_grad,FLAGS.clip_grad), v] for g, v in grads_and_vars]
    train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        #模型的训练
        num_inter = int(len(train_sen_len_list) / FLAGS.batch_size)
        for epoch in range(FLAGS.num_epochs):
            for i in range(num_inter):
                start=i*FLAGS.batch_size
                end=(i+1)*FLAGS.batch_size
                feed_dict={x:train_sen_index_list[start:end],y:train_labels[start:end],sequence_lengths:train_sen_len_list[start:end],keep_prob:FLAGS.keep_prob}
                if i%10==0:
                    train_loss=sess.run(loss,feed_dict={x:train_sen_index_list[start:end],y:train_labels[start:end],
                                              sequence_lengths:train_sen_len_list[start:end],keep_prob:FLAGS.keep_prob})
                    print("epoch:%d step:%d loss is:%s" % (epoch+1,i,train_loss))
                sess.run(train_op,feed_dict=feed_dict)
        #对测试集进行测试
        logits, transition_params=sess.run([logits,transition_params],feed_dict={
            x:test_sen_index_list,y:test_labels,
            sequence_lengths:test_sen_len_list,keep_prob:1.0
        })
        label_list = []
        for logit, seq_len in zip(logits, test_sen_len_list):
            #viterbi_decode通俗一点,作用就是返回最好的标签序列.这个函数只能够在测试时使用,在tensorflow外部解码
            #viterbi: 一个形状为[seq_len] 显示了最高分的标签索引的列表.
            #viterbi_score: 序列对应的概率值
            #这是解码的过程，利用维比特算法结合概率转移矩阵求得最大的可能标注概率
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        #索引向标签的转换
        label2tag={}
        for label,tag in tag2label.items():
            label2tag[tag]=label
        tags_list=[]
        for labels in label_list:
            tags=[]
            for i in labels:
                tags.append(label2tag[i])
            tags_list.append(tags)
        #计算精度
        accuracy_num=0
        sum_num=0
        for pre_tags,test_tags in zip(tags_list,test_tags_list):
            sum_num=sum_num+len(test_tags)
            for pre_tag,test_tag in zip(pre_tags,test_tags):
                if pre_tag==test_tag:
                    accuracy_num=accuracy_num+1
        print(accuracy_num/sum_num)
if __name__ == '__main__':
    backward_propagation()