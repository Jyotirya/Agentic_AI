from transformers import pipeline
summarizer = pipeline("summarization")

text = "Life is a journey filled with both challenges and opportunities, and it is up to each individual to find meaning and purpose along the way. Embracing change and learning from failures can lead to personal growth and resilience. It is important to cultivate gratitude, cherish relationships, and practice kindness, as these bring true fulfillment. Balancing ambition with contentment allows us to strive for our goals while appreciating the present moment. Ultimately, living life fully means being true to oneself, staying curious, and making a positive impact on the world around us."

summary = summarizer(text, max_length=50,min_length=25)

print("Length of Original Text:", len(text))
print("Length of Summary:", len(summary[0]['summary_text']))
print("Summary:", summary[0]['summary_text']) 