import streamlit as st
import json
import os
import urllib.parse

# files
INPUT_FILE = "manual_annotation.json"
REVIEWED_FILE = "manual_annotation_reviewed.json"

if os.path.exists(REVIEWED_FILE):
    with open(REVIEWED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(REVIEWED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

total = len(data)

if "index" not in st.session_state:
    for i, entry in enumerate(data):
        if "match_llm" not in entry:
            st.session_state.index = i
            break
    else:
        st.session_state.index = total

current_index = st.session_state.index

if current_index >= total:
    st.success("✅ Review completed")
    st.stop()

# current
track = data[current_index]
artist = track["artist"]
title = track["title"]
full_title = f"{artist} - {title}"
query = urllib.parse.quote_plus(f"{artist} {title}")

# headers
st.markdown(f"# 🎵 {full_title}")
st.markdown(f"🔢 Review {current_index + 1} of {total}")

# copy button
st.code(full_title)
copy_code = f"""
<script>
function copyText() {{
  navigator.clipboard.writeText("{full_title}");
}}
</script>
<button onclick="copyText()">📋 Copy</button>
"""
st.components.v1.html(copy_code, height=40)

st.divider()

# search on Google
st.markdown("### 🔍 Search on Google")
google_url = f"https://www.google.com/search?q={query}"
st.markdown(f"[🔗 Search on Google]({google_url})")

if st.button("🚫 None of the videos match"):
    track["match_llm"] = False
    with open(REVIEWED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    st.session_state.index += 1
    st.rerun()

# manual link
st.markdown("### 🎯 Do you want to enter a YouTube video manually?")
manual_link = st.text_input("Paste the YouTube link here if you found a better one:")

if manual_link:
    if st.button("✅ Use this link as manual match"):
        track["match_llm"] = True
        track["manual_youtube_link"] = manual_link.strip()
        with open(REVIEWED_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        st.session_state.index += 1
        st.rerun()
