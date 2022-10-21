from flask import Flask,render_template,request
import pickle
#import torch

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    try:
        #device = torch.device('cpu')
        input = request.form.get('inp')
        #len= request.form.get('num')
        model = pickle.load(open('t5model.pkl','rb'))
        tokenizer=pickle.load(open('t5tokenizer.pkl','rb'))
        
        #preprocess_text = input.strip().replace("\n","")
        #t5_prepared_Text = "summarize: "+preprocess_text
        
        tokenized_text = tokenizer.encode(input, return_tensors="pt")#.to(device)
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
