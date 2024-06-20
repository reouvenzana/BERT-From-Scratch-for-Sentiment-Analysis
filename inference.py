import torch
from transformers import BertTokenizer, BertForSequenceClassification
from accelerate import Accelerator

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model.load_state_dict(torch.load('./results/model_epoch_4.pth'))  # Modifier le chemin si nécessaire

accelerator = Accelerator()
model = accelerator.prepare(model)

model.eval()

def predict_review(review_text):
    # Tokenizer la critique
    inputs = tokenizer(review_text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(accelerator.device)
    with torch.no_grad():
        
        outputs = model(**inputs) # Making a prediction

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1) # Obtain the class with the highest probability
    
    prediction = torch.argmax(probs, dim=-1).cpu().item()
    # Déterminer le sentiment
    sentiment = 'POSITIVE' if prediction == 1 else 'NEGATIVE'
    return sentiment


positive_reviews = [
    "This movie was an absolute masterpiece! The acting was superb and the storyline captivating.",
    "I loved every minute of this film. The characters were well-developed and the plot was intriguing.",
    "A truly inspiring movie with a powerful message. The performances were outstanding.",
    "The cinematography was breathtaking and the direction was flawless. Highly recommended!",
    "An incredible film that touched my heart. The actors did a phenomenal job.",
    "One of the best movies I've seen in a long time. The script was brilliant and the execution perfect.",
    "A beautiful and emotional journey. The film was expertly crafted and deeply moving.",
    "Fantastic! The visuals were stunning and the soundtrack was perfect. A must-watch!",
    "An unforgettable experience. The storytelling was on point and the performances were top-notch.",
    "Absolutely loved it! The film was thought-provoking and the acting was brilliant."
]

negative_reviews = [
    "This movie was a complete waste of time. The plot was boring and the acting was terrible.",
    "I couldn't get through the whole film. The characters were uninteresting and the story was dull.",
    "The movie was poorly made and lacked any real substance. Very disappointing.",
    "A total disaster. The direction was awful and the script was even worse.",
    "One of the worst movies I've ever seen. The acting was atrocious and the plot made no sense.",
    "I regret watching this film. It was a complete mess from start to finish.",
    "The movie was incredibly slow and the storyline was unengaging. I wouldn't recommend it.",
    "Terrible film. The pacing was off and the performances were lackluster.",
    "An absolute flop. The movie failed to deliver on all fronts.",
    "I found this film to be extremely disappointing. The plot was convoluted and the acting was subpar."
]

# Tester les critiques positives
print("Testing Positive Reviews:")
for review in positive_reviews:
    predicted_sentiment = predict_review(review)
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {predicted_sentiment}\n")

# Tester les critiques négatives
print("Testing Negative Reviews:")
for review in negative_reviews:
    predicted_sentiment = predict_review(review)
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {predicted_sentiment}\n")