from flask import Flask,render_template,request
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    try:
        device = torch.device('cpu')
        input = request.form.get('inp')
        preprocess_text = input.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
        # summmarize 
        summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=15,
                                    max_length=30,
                                    early_stopping=True)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return render_template("res.html",output=output,input=input)
                   
    except Exception as er:
        print("-----exception--->",er)
        return 'exception'        


if __name__ == "__main__":
    app.run(debug=True)
