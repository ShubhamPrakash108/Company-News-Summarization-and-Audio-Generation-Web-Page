from flask import Flask, request, jsonify, render_template, send_from_directory
import json
import os
from utils import save_company_news
from utils import sentiment_analysis_model
from utils import news_summarization, audio_output, Topic_finder, GEMINI_LLM_COMPARISON
from collections import Counter
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import gc
import torch

app = Flask(__name__, static_folder="static")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_company():
    company_name = request.form.get('company_name')
    
    if not company_name:
        return jsonify({"error": "Please enter a company name."}), 400
    
    try:
        os.makedirs("Company", exist_ok=True)
        os.makedirs("audios", exist_ok=True)
        
        file_path = save_company_news(company_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "Failed to fetch news. Try again."}), 400
        
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        for article in data:
            topics = Topic_finder(article['title'])
            
            sentiment = sentiment_analysis_model(article['content'])
            article["sentiment"] = sentiment['sentiment']
            
            del sentiment
            gc.collect()
            
            summary = news_summarization(article["content"])
            article["summary"] = summary
            
            article["topics"] = topics
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
        
        sentiment_counts = Counter(article["sentiment"] for article in data if "sentiment" in article)
        
        output_data = {
            "Articles": data, 
            "Comparative_Sentiment_Score": {  
                "Sentiment_Counts": {
                    "Positive": sentiment_counts.get("Positive", 0),
                    "Negative": sentiment_counts.get("Negative", 0),
                    "Neutral": sentiment_counts.get("Neutral", 0),
                }
            }
        }
        
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=4)
        
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        articles = data["Articles"]
        comparative_score = data["Comparative_Sentiment_Score"]
        
        comparisons = []
        hindi_text = ""
        
        audio_num = 1
        
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                x = articles[i]['title']
                y = articles[j]['title']
                result = GEMINI_LLM_COMPARISON(f"Compare {x} and {y}")
                result = result.replace("*", "")
                hindi_translation = GoogleTranslator(source="en", target="hi").translate(result)
                hindi_text = hindi_text + hindi_translation
                audio_output(hindi_translation, audio_num)
                comparisons.append(result)
                audio_num = audio_num + 1
        
        output_data = {
            "Articles": articles, 
            "Comparative_Sentiment_Score": comparative_score,
            "Comparison_through_articles": comparisons
        }
        
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=4)
        
        with open("translated_text.txt", "w", encoding="utf-8") as file:
            file.write(hindi_text)
        
        audio_folder = "audios"
        
        audio_files = sorted(
            [f for f in os.listdir(audio_folder) if f.endswith(".wav")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        
        final_audio = AudioSegment.empty()
        
        for file in audio_files:
            file_path = os.path.join(audio_folder, file)
            audio = AudioSegment.from_wav(file_path)
            final_audio += audio
        
        output_file = "merged_audio.wav"
        final_audio.export(output_file, format="wav")
        
        with open(f"Company/{company_name}.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        
        topic_counts = Counter(topic for article in data["Articles"] for topic in article["topics"])
        
        common_topics = {topic for topic, count in topic_counts.items() if count > 1}
        unique_topics = [
            {
                "Article": article["title"],
                "Unique Topics": list(set(article["topics"]) - common_topics)
            }
            for article in data["Articles"]
        ]
        
        data["Comparative_Sentiment_Score"]["Topic_Overlap"] = {
            "Common Topics": list(common_topics),
            "Unique Topics per Article": unique_topics
        }
        
        with open(f"Company/{company_name}.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        
        return jsonify({
            "success": True,
            "company_name": company_name,
            "data": data
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_data/<company_name>')
def get_data(company_name):
    try:
        with open(f"Company/{company_name}.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(".", filename)

if __name__ == '__main__':
    app.run(debug=True)