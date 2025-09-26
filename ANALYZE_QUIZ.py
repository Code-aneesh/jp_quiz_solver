#!/usr/bin/env python3
"""
DIRECT JAPANESE QUIZ ANALYZER
============================
Analyzes the specific quiz questions from your image
"""

import google.generativeai as genai
import os

# Configure AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)

def analyze_quiz():
    """Analyze the specific Japanese quiz from the image"""
    
    # The exact quiz content from your image
    quiz_text = """
問題II ______のことば はどう かきますか。1234からいちばんいいものを
ひとつえらびなさい。

問1・ほほと やまに のぼりました。
(16) ほほ  1.娘  2.什  3.奶  4.母
(17) やま  1.上  2.山  3.止  4.占

問2・こんしゅうは てんきが よかった。
(18) こんしゅう  1.今週  2.今過  3.令週  4.令過
(19) てんき  1.天気  2.天汽  3.矢気  4.矢汽

問3・その ちいさい かれんだーを ください。
(20) ちいさい  1.小い  2.小さい  3.少い  4.少さい
(21) かれんだー  1.カトンクー  2.カトンダー  3.カレンクー  4.カレンダー

問4・ひがしの そらが きれいです。
(22) ひがし  1.束  2.東  3.南  4.北
(23) そら  1.川  2.池  3.空  4.風

問5・むいかの ごごに あいましょう。
(24) むいか  1.九日  2.三日  3.六日  4.五日
(25) ごご  1.午役  2.牛役  3.午後  4.牛後
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
You are an expert Japanese language teacher and JLPT specialist. Analyze this Japanese quiz and provide the correct answers with detailed explanations.

Quiz:
{quiz_text}

This is a hiragana to kanji/katakana conversion test. For each question, provide:

🎯 QUESTION [number]: [Brief description of what's being tested]
✅ CORRECT ANSWER: [Number and the correct option]
📝 EXPLANATION: [Why this is correct, including meaning and usage]
🔤 READING: [How to pronounce/read the term]

Please be very specific about which numbered choice is correct (1, 2, 3, or 4) and explain why the other options are wrong.
"""
        
        response = model.generate_content(prompt)
        
        print("🎯 JAPANESE QUIZ ANALYSIS")
        print("=" * 60)
        print(response.text)
        
        return response.text
        
    except Exception as e:
        error_msg = f"❌ Analysis Error: {e}"
        print(error_msg)
        return error_msg

def main():
    print("🎯 Starting Direct Quiz Analysis...")
    
    if GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
        print("❌ GEMINI_API_KEY not set! Please set your API key.")
        print("Set it as an environment variable: GEMINI_API_KEY=your_key_here")
        return
    
    print("✅ Analyzing your Japanese quiz...\n")
    analyze_quiz()

if __name__ == "__main__":
    main()
