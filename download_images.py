#!/usr/bin/env python

"""
Download training images for Spider-Man recognizer.
Uses Google Custom Search API to find images.
Downloads in 2:1 ratio of not-Spider-Man to Spider-Man by default.

Tracks downloaded images in a manifest (download_manifest.json) so that
re-runs only download new images.

Requirements:
  - Set environment variables GOOGLE_API_KEY and GOOGLE_CSE_ID
  - Create a Custom Search Engine at https://programmablesearchengine.google.com/
  - Enable the Custom Search API in Google Cloud Console
"""

import os
import sys
import json
import argparse
import hashlib
import requests
from pathlib import Path
from googleapiclient.discovery import build

DOWNLOAD_CACHE_PATH = Path("dataset", "download_cache.json")

SPIDERMAN_QUERIES_TRAIN = [
    "spider-man superhero",
    "spider-man movie still",
    "spider-man comic",
    "spider-man costume",
    "spider-man action pose",
]

SPIDERMAN_QUERIES_VAL = [
    "spider-man wallpaper",
    "spider-man fan art",
    "spider-man screenshot",
    "spider-man poster",
    "spider-man cosplay",
]

NOT_SPIDERMAN_QUERIES_TRAIN = [
    "the hulk relistic",
    "superman relistic",
    "landscape photography",
    "city street photo",
    "cat photo",
    "dog photo",
    "car photo",
    "food photo",
    "people walking",
    "forest nature",
    "ocean beach",
    "building architecture",
]

NOT_SPIDERMAN_QUERIES_VAL = [
    "the hulk movie still",
    "superman movie still",
    "mountain scenery",
    "bird photo",
    "flower garden",
    "sunset sky",
    "bicycle photo",
    "coffee shop interior",
    "river waterfall",
    "snow winter scene",
    "train station",
    "bookshelf library",
]


def load_cache():
    """Load the download cache tracking previously downloaded images."""
    if DOWNLOAD_CACHE_PATH.exists():
        with open(DOWNLOAD_CACHE_PATH, "r") as f:
            return set(json.load(f))
    return set()


def save_cache(cache):
    """Save the download cache."""
    with open(DOWNLOAD_CACHE_PATH, "w") as f:
        json.dump(sorted(cache), f, indent=2)


def search_images(api_key, cse_id, query, num_results=10, start=1):
    """Search for images using Google Custom Search API."""
    service = build("customsearch", "v1", developerKey=api_key)
    results = []
    res = (
        service.cse()
        .list(q=query, cx=cse_id, searchType="image", num=num_results, start=start)
        .execute()
    )
    if "items" in res:
        for item in res["items"]:
            results.append(item["link"])
    return results


def download_image(url, save_path):
    """Download a single image from URL."""
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        return True
    except (requests.RequestException, OSError):
        return False


def download_category(api_key, cse_id, queries, output_dir, target_count, cache):
    """Download images for a category until target count is reached."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = set(
        p.name for p in output_dir.iterdir() if p.suffix in (".jpg", ".png")
    )
    downloaded = len(existing_files)

    print(f"  Found {downloaded} existing images in {output_dir}")
    if downloaded >= target_count:
        print(f"  Already at target ({target_count}), skipping.")
        return downloaded

    for query in queries:
        if downloaded >= target_count:
            break
        print(f"  Searching: '{query}'")
        try:
            urls = search_images(api_key, cse_id, query, num_results=50)
        except Exception as e:
            print(f"    Error searching: {e}")
            continue

        for url in urls:
            if downloaded >= target_count:
                break
            # Skip URLs we've already processed (downloaded or failed)
            if url in cache:
                continue
            cache.add(url)

            # Use URL hash as filename to avoid duplicates
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            ext = ".jpg"
            if ".png" in url.lower():
                ext = ".png"
            save_path = output_dir / f"{url_hash}{ext}"
            if save_path.name in existing_files:
                continue
            if download_image(url, save_path):
                downloaded += 1
                existing_files.add(save_path.name)
                print(f"    [{downloaded}/{target_count}] Downloaded: {save_path.name}")
            else:
                print(f"    Failed: {url[:60]}...")

    # Save cache after processing this category
    save_cache(cache)

    print(f"  Total: {downloaded}/{target_count} images")
    return downloaded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download training images for Spider-Man recognizer."
    )
    parser.add_argument(
        "--spiderman-count",
        type=int,
        default=50,
        help="Total Spider-Man images to download (default: 50)",
    )
    parser.add_argument(
        "--not-spiderman-count",
        type=int,
        default=None,
        help="Total not-Spider-Man images to download (default: 2x spiderman-count)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of images for validation (default: 0.2)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "val"],
        default="all",
        help="Which split to download: train, val, or all (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Output dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google API key (default: GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--cse-id",
        type=str,
        default=os.environ.get("GOOGLE_CSE_ID"),
        help="Google CSE ID (default: GOOGLE_CSE_ID env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_key or not args.cse_id:
        print("Error: Set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables,")
        print("       or pass --api-key and --cse-id.")
        sys.exit(1)

    not_spiderman_count = args.not_spiderman_count or args.spiderman_count * 2
    output_dir = Path(args.output_dir)

    # Split between train and val
    train_spiderman = int(args.spiderman_count * (1 - args.val_split))
    val_spiderman = args.spiderman_count - train_spiderman
    train_not_spiderman = int(not_spiderman_count * (1 - args.val_split))
    val_not_spiderman = not_spiderman_count - train_not_spiderman

    cache = load_cache()

    if args.split in ("all", "train"):
        print("=== Downloading Spider-Man images (train) ===")
        download_category(
            args.api_key, args.cse_id,
            SPIDERMAN_QUERIES_TRAIN, output_dir / "train/spiderman", train_spiderman, cache
        )

        print("\n=== Downloading Not-Spider-Man images (train) ===")
        download_category(
            args.api_key, args.cse_id,
            NOT_SPIDERMAN_QUERIES_TRAIN, output_dir / "train/not-spiderman",
            train_not_spiderman, cache
        )

    if args.split in ("all", "val"):
        print("\n=== Downloading Spider-Man images (val) ===")
        download_category(
            args.api_key, args.cse_id,
            SPIDERMAN_QUERIES_VAL, output_dir / "val/spiderman", val_spiderman, cache
        )

        print("\n=== Downloading Not-Spider-Man images (val) ===")
        download_category(
            args.api_key, args.cse_id,
            NOT_SPIDERMAN_QUERIES_VAL, output_dir / "val/not-spiderman",
            val_not_spiderman, cache
        )

    print(f"\nDone! Dataset ready in ./{output_dir}/")
    print(f"Cache saved to {DOWNLOAD_CACHE_PATH}")


if __name__ == "__main__":
    main()
