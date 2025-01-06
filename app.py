import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import shutil

model_name = "minoD/JURAN"

# モデルのロード
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# プロンプトテンプレートの準備
def generate_prompt(F):
    # input キーの代わりに Q と F を使用
    result = f"""### 指示:あなたは企業の面接官です．就活生のエントリーシートを元に質問を行ってください．

### 質問:
{F}

### 回答:
""" # 回答セクションを追加
    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result

# テキスト生成関数の定義
def generate2(F=None, maxTokens=256):
    # 推論
    prompt = generate_prompt(F)
    input_ids = tokenizer(prompt,
                          return_tensors="pt",
                          truncation=True,
                          add_special_tokens=False).input_ids
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=maxTokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.75,
        top_k=40,
        no_repeat_ngram_size=2,
    )
    outputs = outputs[0].tolist()
    decoded = tokenizer.decode(outputs)

    # EOSトークンにヒットしたらデコード完了
    if tokenizer.eos_token_id in outputs:
        eos_index = outputs.index(tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[:eos_index])

        # レスポンス内容のみ抽出
        sentinel = "### 回答:"
        sentinelLoc = decoded.find(sentinel)
        if sentinelLoc >= 0:
            result = decoded[sentinelLoc + len(sentinel):]
            return result.replace("<NL>", "\n")  # <NL>→改行
        else:
            return 'Warning: Expected prompt template to be emitted.  Ignoring output.'
    else:
       return 'Warning: no <eos> detected ignoring output'
    
def inference(input_text):
  return generate2(input_text)


iface = gr.Interface(
    fn=inference,
    inputs=gr.Textbox(lines=5, label="学生時代に打ち込んだこと、研究、ESを入力", placeholder="半導体の研究に打ち込んだ"),
    outputs=gr.Textbox(label="想定される質問"),
    title="JURAN🌺",
    description="predict",
)

if __name__ == "__main__":
    iface.launch(share=True)