import tiktoken
encoding =tiktoken.encoding_for_model("gpt-4o-mini")
Sentence={
    "English":"I am transforming the world",
    "Hindi": "मैं दुनिया को बदल रहा हूँ।",
    "Code":"def add(a,b): return a+b"
}
token_counts={}
for lang, text in Sentence.items():
    tokens=encoding.encode(text)
    token_counts[lang]=len(tokens)
    print(f"\n{lang} Sentence:{text}")
    print("Tokens:",tokens)
    print("Token Count:",len(tokens))