# bert4keras加载CDial-GPT

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout
import tensorflow as tf
from bert4keras.backend import K
from flask import Flask, request, render_template, send_file



app = Flask(__name__)

graph = tf.get_default_graph()
sess = K.get_session()
set_session = K.set_session

config_path = r'GPT_large-tf\gpt_config.json'
checkpoint_path = r'GPT_large-tf\gpt_model.ckpt'
dict_path = r'GPT_large-tf\vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
speakers = [
    tokenizer.token_to_id('[speaker1]'),
    tokenizer.token_to_id('[speaker2]')
]

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='GPT_OpenAI'
)  # 建立模型，加载权重

# model.save("gpt.h5")
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
# print(chatbot.response([u'别爱我没结果', u'你这样会失去我的', u'失去了又能怎样']))
"""
回复是随机的，例如：你还有我 | 那就不要爱我 | 你是不是傻 | 等等。
"""
@app.route("/params", methods=["GET"])
def params():
    with graph.as_default():
        set_session(sess)
        text = request.args.get("text")
        print("用户:",text)

        result = chatbot.response([text])
        print("bot:",result)
    return result

# 上传数据
@app.route("/", methods=["GET"])
def multi_view():
    return send_file('bot2.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)


