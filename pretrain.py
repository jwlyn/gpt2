#! -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator,AutoRegressiveDecoder
from bert4keras.snippets import uniout

from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from keras.models import Model


maxlen = 256
batch_size = 1
steps_per_epoch = 3
epochs = 10000

config_path = 'GPT_large-tf/gpt_config.json'
checkpoint_path = 'GPT_large-tf/gpt_model.ckpt'
dict_path = 'GPT_large-tf/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
speakers = [
    tokenizer.token_to_id('[speaker1]'),
    tokenizer.token_to_id('[speaker2]')
]



class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, text in self.sample(random):
            token_ids = [tokenizer._token_start_id, speakers[0]]
            segment_ids = [tokenizer._token_start_id, speakers[0]]
            for i, text in enumerate(text):
                ids = tokenizer.encode(text)[0][1:-1] + [speakers[(i + 1) % 2]]
                token_ids.extend(ids)
                segment_ids.extend([speakers[i % 2]] * len(ids))
                segment_ids[-1] = speakers[(i + 1) % 2]

            token_ids[-1]= tokenizer._token_end_id
            segment_ids[-1] = speakers[i % 2]

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='GPT_OpenAI'
)  # 建立模型，加载权重

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()

class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        curr_segment_ids = np.zeros_like(output_ids) + token_ids[0, -1]
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def response(self, texts, topk=5):
        token_ids = [tokenizer._token_start_id, speakers[0]]
        segment_ids = [tokenizer._token_start_id, speakers[0]]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:-1] + [speakers[(i + 1) % 2]]
            token_ids.extend(ids)
            segment_ids.extend([speakers[i % 2]] * len(ids))
            segment_ids[-1] = speakers[(i + 1) % 2]
        results = self.random_sample([token_ids, segment_ids], 1, topk)
        return tokenizer.decode(results[0])


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)



class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        # 演示效果
        # just_show()

txts = [["你好","您好","你来自哪","中国"],["1+1","2"],["你来自哪","中国"]]
if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(txts, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')
