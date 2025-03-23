# Company News Summarization and Audio Generation Web Page

This project is a Flask-based web application designed to fetch, summarize, and analyze company-related news articles. It performs sentiment analysis, topic extraction, and audio generation in Hindi.

## Features

- **Fetch Latest News**: Scrapes the web to retrieve news articles related to a specified company.
- **Sentiment Analysis**: Classifies articles into Positive, Neutral, or Negative categories.
- **News Summarization**: Uses NLP models to generate concise summaries.
- **Topic Extraction**: Identifies key topics from articles.
- **Comparative Analysis**: Compares multiple articles and generates insights.
- **Hindi Translation & Audio Generation**: Translates comparisons into Hindi and converts them to speech.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ShubhamPrakash108/Company-News-Summarization-and-Audio-Generation-Web-Page.git
   cd Company-News-Summarization-and-Audio-Generation-Web-Page
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file and adding:
   ```
   GEMINI_API_KEY_1=your_api_key
   GEMINI_API_KEY_2=your_api_key
   GEMINI_API_KEY_3=your_api_key
   ```

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```
2. Open in a browser:
   ```
   http://127.0.0.1:5000/
   ```
3. Enter a company name to analyze its news.

## API Endpoints

- **`/analyze` (POST)** - Fetches and analyzes news for a given company.
- **`/get_data/<company_name>` (GET)** - Retrieves stored news data for a company.
- **`/audio/<filename>` (GET)** - Serves generated audio files.

## Technologies Used

- **Flask** - Web framework
- **BeautifulSoup & Newspaper3k** - Web scraping
- **Transformers & BERTopic** - NLP and topic modeling
- **Torch & Deep Translator** - Sentiment analysis and translation
- **Google Generative AI** - Article comparison
- **Pydub & SoundFile** - Audio processing

## Folder Structure

```
│── Company/           # Stores fetched news data
│── audios/            # Stores generated audio files
│── templates/         # HTML templates
│── app.py             # Flask application
│── utils.py           # Utility functions
│── requirements.txt   # Dependencies
│── .env               # API keys configuration
│── README.md          # Project documentation
```

## Future Enhancements

- Improve topic modeling with advanced LLMs.
- Enhance audio quality using better TTS models.
- Implement real-time news updates.


## License

This project is licensed under the MIT License.

---


