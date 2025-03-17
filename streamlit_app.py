import streamlit as st
import openai
import pandas as pd
import json

# Set your OpenAI API key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Burgo's Headline Comparator", layout="centered", page_icon="🆚")
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

# Build system prompt
def get_system_prompt():
    return """
You are an expert direct response copywriter and copy evaluator.

Your task is to rate and explain multiple headlines for digital ads. You will evaluate them using the following five attributes, each rated from 1 to 10:

1. Clarity – Is the meaning obvious on first read? Avoids confusion or jargon.
2. Emotional Pull – Does it evoke a feeling like excitement, fear, greed, hope, etc.?
3. Curiosity – Does it create a desire to click and learn more?
4. Persuasive Strength – How compelling is the argument or benefit?
5. CTR Potential – Heuristic score of how likely this headline is to drive clicks. This is an estimate based on copywriting principles — not based on real ad data.

Also identify the likely **copywriting framework** used (e.g. AIDA, PAS, None).

### Output Format
Return a JSON array like this:
[
  {
    "headline": "<headline text>",
    "clarity": 8,
    "emotional_pull": 7,
    "curiosity": 9,
    "persuasive_strength": 8,
    "ctr_potential": 9,
    "framework": "AIDA",
    "explanation": "Short explanation of strengths and weaknesses."
  },
  ...
]

Then return a final ranking of headlines by CTR potential:

**Overall Ranking**:
1. "<top headline>"
2. ...
"""

# Build user prompt
def build_prompt(headlines):
    headlines_block = "\n".join([f"{i + 1}. {hl}" for i, hl in enumerate(headlines)])
    return f"Please evaluate the following headlines:\n\n{headlines_block}"

# GPT Prompt to Generate CTA Suggestions

def build_cta_prompt(headline):
    return f"""
You're a seasoned direct response copywriter.

Given this high-performing headline:
"{headline}"

Generate:
1. Three short **subheadlines or lead-ins** (1–2 lines max) that could follow this in an ad or landing page. These should support or expand on the headline's message.
2. Three **CTA button suggestions** (1–4 words each), matching the tone and promise of the headline.

Respond using this exact structure:
---
**Subheadlines:**
- ...
- ...
- ...

**CTA Buttons:**
- ...
- ...
- ...
---
"""

# Parse JSON output into DataFrame
def parse_json_results(json_str):
    try:
        data = json.loads(json_str.split("**Overall Ranking**:")[0])
        df = pd.DataFrame(data)
        df.columns = [col.title().replace("_", " ") for col in df.columns]
        columns_order = ["Headline", "Ctr Potential", "Clarity", "Emotional Pull", "Curiosity", "Persuasive Strength", "Framework"]
        return df[columns_order]
    except Exception as e:
        st.error(f"Failed to parse GPT output: {e}")
        return pd.DataFrame()

# Highlight best-performing row
def highlight_top(df):
    top_idx = df["Ctr Potential"].idxmax()
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
                system_prompt = get_system_prompt()
                prompt = build_prompt(headlines)
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                result = response.choices[0].message.content
                df = parse_json_results(result)

                if not df.empty:
                    st.markdown("---")
                    top_headline = df.loc[df["Ctr Potential"].idxmax(), "Headline"]
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
    try:
        # Split into JSON and ranking
        json_part, *rest = result.split("**Overall Ranking**:")
        data = json.loads(json_part)

        for item in data:
            st.markdown(f"""\
**Headline:** \"{item['headline']}\"  
- **Clarity**: {item['clarity']}/10  
- **Emotional Pull**: {item['emotional_pull']}/10  
- **Curiosity**: {item['curiosity']}/10  
- **Persuasive Strength**: {item['persuasive_strength']}/10  
- **CTR Potential**: {item['ctr_potential']}/10  
- **Framework**: {item['framework']}  
- **Explanation**: {item['explanation']}
---
""")

        if rest:
            st.markdown("**Overall Ranking:**")
            st.markdown(rest[0].strip())

    except Exception as e:
        st.error(f"Error displaying full GPT analysis: {e}")

            except Exception as e:
                st.error(f"Error generating results: {e}")
