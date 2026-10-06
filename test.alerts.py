import google.generativeai as genai

genai.configure(api_key="AQ.Ab8RN6Kb7p0Q6l0VYHUaZnY0JS3--KyaZMzDVL9Gbiyo7TSU9A")

print("Available Models for Generation:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)