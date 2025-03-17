import streamlit as st
import openai
import os
import re
import pandas as pd

# Set your OpenAI API key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Burgo's Headline Comparator", layout="centered", page_icon="🧠")
st.title("🧠 Burgo's Headline Comparator Tool")

# Introductory paragraph
st.markdown("""
### ℹ️ About This Tool

This is an early prototype built using a **heuristic model** powered by GPT-4o. That means it's drawing on expert-level language knowledge (not live ad data) to simulate how a skilled copywriter might evaluate headlines.

At a later stage — once we’ve built out our **Creative Intelligence Platform™** (watch this space!) — we plan to train this tool on **real ad performance data**, using a machine learning approach to predict actual CTR with much greater accuracy.

For now, this MVP is designed to help you:
- Compare up to **5 headlines** side by side
- See **attribute scores** like Clarity, Curiosity, Emotional Pull, etc.
- Identify the **top-rated headline** for CTR potential
- View **GPT-generated rationale and ranking**
- Expand to see **full scoring analysis** and **CTA-style suggestions** for your top headline
""")

st.markdown("Compare up to 5 headlines to see which is most persuasive.")

# Input form
with st.form("headline_form"):
    headlines = []
    for i in range(1, 6):
        headline = st.text_input(f"Headline {i}", key=f"headline_{i}")
        if headline:
            headlines.append(headline)

    submitted = st.form_submit_button("Compare Headlines")

# GPT Prompt to Score Headlines
def build_prompt(headlines):
    headlines_block = "\n".join([f"{i + 1}. {hl}" for i, hl in enumerate(headlines)])
    return f"""
You are a seasoned copywriting expert with deep experience in direct response marketing and high-converting digital ads. You're helping a team evaluate multiple headline options.

Your job is to analyze each headline for the following attributes:
1. **Clarity**
2. **Emotional Pull**
3. **Curiosity**
4. **Persuasive Strength**
5. **CTR Potential**

Please rate each from 1 to 10 and explain briefly. Then, identify the **likely persuasion framework** (e.g. AIDA, PAS, None).

Headlines to evaluate:
{headlines_block}

Return your answer in this format:

---
**Headline**: "<headline>"

- Clarity: X/10 — <why>
- Emotional Pull: X/10 — <why>
- Curiosity: X/10 — <why>
- Persuasive Strength: X/10 — <why>
- CTR Potential: X/10 — <why>
- Framework: <e.g., AIDA, PAS, None>

---

Then end with:
**Overall Ranking**:
1. "<headline>"
2. ...
"""

# GPT Prompt to Generate CTA Suggestions
def build_cta_prompt(headline):
    return f"""
You are an expert direct response copywriter. Based on the following high-performing headline:

\"{headline}\"

Generate:
1. Three persuasive subheadlines or lead-in sentences (1–2 lines max) that could follow this in an ad or landing page.
2. Three short CTA button suggestions (1–4 words each) that match the tone and promise of the headline.

Respond using this structure:

---
**Subheadlines:**
- <subheadline 1>
- <subheadline 2>
- <subheadline 3>

**CTA Buttons:**
- <button 1>
- <button 2>
- <button 3>
---
"""

# Parse GPT results into DataFrame
def parse_results(result):
    pattern = r'\*\*Headline\*\*: \"(.*?)\".*?- Clarity: (\d+)/10.*?- Emotional Pull: (\d+)/10.*?- Curiosity: (\d+)/10.*?- Persuasive Strength: (\d+)/10.*?- CTR Potential: (\d+)/10.*?- Framework: (.*?)\n'
    matches = re.findall(pattern, result, re.DOTALL)

    data = []
    for m in matches:
        data.append({
            "Headline": m[0],
            "Clarity": int(m[1]),
            "Emotional Pull": int(m[2]),
            "Curiosity": int(m[3]),
            "Persuasive Strength": int(m[4]),
            "CTR Potential": int(m[5]),
            "Framework": m[6].strip()
        })
    df = pd.DataFrame(data)
    columns_order = ["Headline", "CTR Potential", "Clarity", "Emotional Pull", "Curiosity", "Persuasive Strength", "Framework"]
    return df[columns_order]

# Highlight best-performing row
def highlight_top(df):
    top_idx = df["CTR Potential"].idxmax()
    def highlight_row(row):
        return ["background-color: #ffd700" if row.name == top_idx else "" for _ in row]
    return df.style.apply(highlight_row, axis=1)

# On submission
if submitted:
    if len(headlines) < 2:
        st.warning("Please enter at least 2 headlines for comparison.")
    else:
        with st.spinner("Evaluating headlines with GPT-4o..."):
            try:
                prompt = build_prompt(headlines)
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert copywriting analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                result = response.choices[0].message.content
                df = parse_results(result)

                st.markdown("---")
                top_headline = df.loc[df["CTR Potential"].idxmax(), "Headline"]
                st.success(f"🏆 **Top Headline:** _{top_headline}_")

                st.markdown("### 📊 Headline Scorecard")
                st.dataframe(highlight_top(df), use_container_width=True)

                # Generate CTA Suggestions
                with st.spinner("Generating CTA suggestions for the top headline..."):
                    try:
                        cta_prompt = build_cta_prompt(top_headline)
                        cta_response = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a top-tier direct response copywriter."},
                                {"role": "user", "content": cta_prompt}
                            ],
                            temperature=0.7
                        )
                        cta_output = cta_response.choices[0].message.content
                        with st.expander("💬 Suggested Subheads & CTA Buttons"):
                            st.markdown(cta_output)
                    except Exception as e:
                        st.error(f"Error generating CTA suggestions: {e}")

                with st.expander("📝 Full GPT Analysis"):
                    st.markdown(result)

            except Exception as e:
                st.error(f"Error generating results: {e}")
