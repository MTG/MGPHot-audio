import csv
import json
import time
import ast
import yt_dlp
from tqdm import tqdm
import os


def search_youtube(query, max_results=5):
    """Search on Youtube using yt_dlp and return top 5 results."""
    ydl_opts = {
        'quiet': True,
        'default_search': 'ytsearch',
        'noplaylist': True,
        'skip_download': True,
    }

    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(f'ytsearch{max_results}:{query}', download=False)
            if 'entries' in result:
                for video in result['entries']:
                    videos.append({
                        "link": f"https://www.youtube.com/watch?v={video.get('id', '')}",
                        "title": video.get("title", ""),
                        "views": video.get("view_count", "N/A"),
                        "description": video.get("description", "")
                    })
        except Exception as e:
            print(f"Error en la búsqueda de YouTube para '{query}': {e}")
    return videos


def enrich_dataset_with_youtube(tsv_file, output_file):
    enriched_data = []
    processed_ids = set()

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                enriched_data = json.load(f)
                processed_ids = {entry["mgphot_track_id"] for entry in enriched_data}
            except json.JSONDecodeError:
                print(f"Warning: file {output_file} is corrupted or empty. Starting fresh.")

    with open(tsv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in tqdm(reader):
            track_id = row["mgphot_track_id"]
            if track_id in processed_ids:
                print(f"Ya procesado: {track_id}. Skip.")
                continue

            year = row["year"]
            artist = row["artist"]
            title = row["title"]
            query = f"{year} {artist} {title}"

            try:
                row["gene_values"] = list(map(float, ast.literal_eval(row["gene_values"])))
            except Exception as e:
                print(f"Error for '{title}': {e}")
                row["gene_values"] = []

            print(f"Searching in YouTube: {query}")
            row["youtube"] = search_youtube(query)
            enriched_data.append(row)
            processed_ids.add(track_id)

            # Guardar JSON tras cada entrada
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(enriched_data, f, indent=4, ensure_ascii=False)

            time.sleep(1)

    print(f"All the data saved on {output_file}")


if __name__ == "__main__":
    enrich_dataset_with_youtube("mgphot_gene_values.tsv", "dataset_with_youtube.json")
