import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from newspaper import Article
from html import unescape
from transformers import pipeline,VitsModel, AutoTokenizer
import torch
import soundfile as sf
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import random

def clean_text(text):
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text.strip()

def search_news(company_name, num_articles=10):
    query = f"{company_name} news".replace(' ', '+')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    search_url = f"https://www.google.com/search?q={query}&tbm=nws"
    
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_links = []
        news_divs = soup.find_all('div', class_='SoaBEf')
        
        for div in news_divs:
            link_tag = div.find('a')
            if link_tag:
                href = link_tag.get('href')
                if href.startswith('/url?q='):
                    url = href.split('/url?q=')[1].split('&sa=')[0]
                    news_links.append(url)
                elif href.startswith('http'):
                    news_links.append(href)
        
        return news_links
    except Exception as e:
        print(f"Error searching for news: {str(e)}")
        return []

def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        if not article.text.strip():
            raise ValueError("Empty article content")
        
        return {
            "title": clean_text(article.title),
            "content": clean_text(article.text),
            "url": url
        }
    except Exception as e:
        print(f"Skipping article {url} due to error: {str(e)}")
        return None

def save_company_news(company_name, num_articles=10):
    news_urls = search_news(company_name)
    articles = []
    
    for url in news_urls:
        if len(articles) >= num_articles:
            break
        
        article_data = extract_article_content(url)
        if article_data:
            articles.append(article_data)
        
        time.sleep(1)
    
    while len(articles) < num_articles:
        additional_urls = search_news(company_name, num_articles=10)
        for url in additional_urls:
            if len(articles) >= num_articles:
                break
            article_data = extract_article_content(url)
            if article_data:
                articles.append(article_data)
            time.sleep(1)
    
    os.makedirs("Company", exist_ok=True)
    file_path = os.path.join("Company", f"{company_name}.json")
    
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(articles, json_file, ensure_ascii=False, indent=4)
    
    return file_path

def sentiment_analysis_model(text):
    text = text[:510]
    classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    result = classifier(text)[0]  
    label_mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    sentiment = label_mapping.get(result["label"], "Unknown")  
    # print({"sentiment": sentiment, "score": result["score"]})  
    return {"sentiment": sentiment}

def news_summarization(ARTICLE):
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    summary = summarizer(ARTICLE, max_length=57)  
    return summary[0]['summary_text']

# def audio_output(text):
#     model = VitsModel.from_pretrained("facebook/mms-tts-hin")
#     tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
#     inputs = tokenizer(text, return_tensors="pt")
#     with torch.no_grad():
#         output = model(**inputs).waveform
#         waveform = output.squeeze().cpu().numpy()
#         sample_rate = 16000  
#         sf.write("output.wav", waveform, sample_rate)

# def audio_output(text, number=1):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     try:
#         model = VitsModel.from_pretrained("facebook/mms-tts-hin").to(device)
#         tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
        
#         inputs = tokenizer(text, return_tensors="pt").to(device)
        
#         with torch.no_grad():
#             output = model(**inputs).waveform
#             waveform = output.squeeze().cpu().numpy()
        
#         sample_rate = 16000  
#         output_file = f"/audios/file_number_{number}.wav"
#         sf.write(output_file, waveform, sample_rate)
#         if device == "cuda":
#             torch.cuda.empty_cache()
            
#         del model
#         del inputs
#         del output
#         del waveform
        
#     except Exception as e:
#         print(f"Error generating audio: {str(e)}")
import os
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

def audio_output(text, number=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = VitsModel.from_pretrained("facebook/mms-tts-hin").to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**inputs).waveform
            waveform = output.squeeze().cpu().numpy()

        sample_rate = 16000  
        
        # Ensure the directory exists
        output_dir = "audios"  # Use relative path
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"file_number_{number}.wav")
        sf.write(output_file, waveform, sample_rate)

        if device == "cuda":
            torch.cuda.empty_cache()

        del model, tokenizer, inputs, output, waveform

    except Exception as e:
        print(f"Error generating audio: {str(e)}")



def Topic_finder(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    topic_model = BERTopic.load("ctam8736/bertopic-20-newsgroups")
    topic_model.embedding_model = embedding_model
    embeddings = embedding_model.encode([text]) 
    topic, _ = topic_model.transform([text], embeddings=embeddings)
    words = topic_model.get_topic(topic[0])
    related_words = [word for word, _ in words]
    return related_words

load_dotenv()

def GEMINI_LLM_COMPARISON(text):
    api_keys = [
        os.getenv("GEMINI_API_KEY_1"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3")
    ]
    api_keys = [key for key in api_keys if key]  

    if not api_keys:
        raise ValueError("No valid API keys found!")

    genai.configure(api_key=random.choice(api_keys))  

    model = genai.GenerativeModel("gemini-2.0-flash-lite")  

    system_instruction = "Compare the following two article headings and generate a clear conclusion in minimum words and in best possible way."

    full_prompt = f"{system_instruction}\n\n{text}"

    conversation = [
        {"role": "user", "parts": [{"text": full_prompt}]}
    ]

    response = model.generate_content(conversation)

    return response.text