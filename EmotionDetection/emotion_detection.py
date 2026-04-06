"""
Emotion detection module using Watson NLP API.
"""

import json
import requests


def emotion_detector(text_to_analyse):
    """
    Analyze emotions of the given text.
    Handles blank input and API errors.

    Args:
        text_to_analyse (str): Input text

    Returns:
        dict: Emotion scores + dominant emotion
    """

    # Handle blank input early
    if text_to_analyse is None or text_to_analyse.strip() == "":
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    myobj = {
        "raw_document": {
            "text": text_to_analyse
        }
    }

    try:
        response = requests.post(url, json=myobj, headers=headers, timeout=5)

        # ✅ Handle 400 status code explicitly
        if response.status_code == 400:
            return {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None
            }

        if response.status_code != 200:
            return {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None
            }

        formatted_response = json.loads(response.text)

        emotions = formatted_response['emotionPredictions'][0]['emotion']

        anger = emotions['anger']
        disgust = emotions['disgust']
        fear = emotions['fear']
        joy = emotions['joy']
        sadness = emotions['sadness']

        dominant_emotion = max(emotions, key=emotions.get)

        return {
            'anger': anger,
            'disgust': disgust,
            'fear': fear,
            'joy': joy,
            'sadness': sadness,
            'dominant_emotion': dominant_emotion
        }

    except Exception:
        # ✅ On any failure, return None values
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }