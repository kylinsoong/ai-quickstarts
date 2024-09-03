from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os

class colors:
    green = '\033[92m'
    blue = '\033[94m'
    red = '\033[31m'
    yellow = '\033[33m'
    reset = '\033[0m'


cog_endpoint = "ENDPOINT"
cog_key = "KEY"
text_analytics_client = TextAnalyticsClient(endpoint=cog_endpoint, credential=AzureKeyCredential(cog_key))


# Change the lines below in quotes to try your own reviews! 
review1 = ["我3月12日在这家餐厅吃饭，只能说一句：哇！食物太棒了，我99%会再来一次！"]
review2 = ["我5月17日来这里吃饭，体验很糟糕。食物上来时是冷的，服务员态度也不太好，我根本不喜欢这里。我不会再来了。"]
review3 = ["这家西餐厅已经很多年了，第一次来，感觉分量很大，太适合大口吃肉，大口喝酒的。"]



def text_analytics(review):
    sentiment_analysis = text_analytics_client.analyze_sentiment(documents=review)
    print("\n-----Sentiment Analysis-----")
    print(f"{colors.blue}The sentence to analyze: {colors.reset}" + str(review))
    for result in sentiment_analysis:
        print(f"{colors.green}Sentiment: {colors.reset}" + result.sentiment)
        print(f"{colors.green}Confidence: {colors.reset}" + str(result.confidence_scores))
    print("----------\n")

    input("Press Enter to continue to key phrases...\n")

    print("\n-----Key Phrases-----")
    print(f"{colors.blue}The sentence to analyze:  {colors.reset}" + str(review))
    key_phrase_analysis = text_analytics_client.extract_key_phrases(documents=review)
    for result in key_phrase_analysis:
        print(f"{colors.green}Key Phrases: {colors.reset}" + str(result.key_phrases))
    print("----------\n")

    input("Press Enter to continue to entities...\n")

    print("\n-----Entities-----")
    print(f"{colors.blue}The sentence to analyze:  {colors.reset}" + str(review))
    entity_analysis = text_analytics_client.recognize_entities(documents=review)
    for result in entity_analysis:
        for entity in result.entities:
            print(f"{colors.green}Entity:{colors.reset} {entity.text:<30}", 
            f" {colors.yellow}Category:{colors.reset} {entity.category:<15}", 
            f" {colors.red}Confidence:{colors.reset} {entity.confidence_score:<4}")
    print("----------\n")

    input("Press Enter to continue...\n")
    print('\033c')


text_analytics(review1)

text_analytics(review2)

text_analytics(review3)

input("Press Enter to exit...\n")
