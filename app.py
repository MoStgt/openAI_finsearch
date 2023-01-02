from flask import Flask, request, render_template
import openai
import pandas as pd
import numpy as np
from config import OPENAI_API_KEY
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity




app = Flask(__name__)

openai.api_key=OPENAI_API_KEY

@app.route('/static/<path:filename>')
def serve_static(filename):
  return app.send_static_file(filename)

@app.route('/')
def search_form():
    return render_template('search_form.html')

@app.route('/search')
def search():
    query = request.args.get('query')
    
    search_term_vector = get_embedding(query, engine="text-embedding-ada-002")
    
    
    df = pd.read_csv('earnings-embeddings.csv')
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    
    sorted_by_similarity = df.sort_values("similarities", ascending=False).head(3)
    results = sorted_by_similarity['text'].values.tolist()
    
    # Render the search results template, passing in the search query and results
    return render_template('search_results.html', query=query, results=results)

if __name__ == '__main__':
    app.config['DEBUG']=True
    app.run()
