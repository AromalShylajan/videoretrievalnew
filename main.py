import clip
import torch
import math
import numpy as np
import extract_frames
import streamlit as st

# Load the open CLIP model
@st.cache_resource
def loadCLIP():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device)
  return device,model, preprocess
def find(query):
  device,model, preprocess = loadCLIP()
  video_frames = extract_frames.extract('video/video.mp4')
  # You can try tuning the batch size for very large videos, but it should usually be OK
  batch_size = 256
  batches = math.ceil(len(video_frames) / batch_size)

  # The encoded features will bs stored in video_features
  video_features = torch.empty([0, 512], dtype=torch.float16).to(device)

  # Process each batch
  for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")

    # Get the relevant frames
    batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
    
    # Preprocess the images for the batch
    batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
    
    # Encode with CLIP and normalize
    with torch.no_grad():
      batch_features = model.encode_image(batch_preprocessed)
      batch_features /= batch_features.norm(dim=-1, keepdim=True)

    # Append the batch to the list containing all features
    video_features = torch.cat((video_features, batch_features))

  # Print some stats
  print(f"Features: {video_features.shape}")



  def search_video(search_query, display_heatmap=True, display_results_count=3):

    # Encode and normalize the search query using CLIP
    with torch.no_grad():
      text_features = model.encode_text(clip.tokenize(search_query).to(device))
      text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute the similarity between the search query and each frame using the Cosine similarity
    similarities = (100.0 * video_features @ text_features.T)
    values, best_photo_idx = similarities.topk(display_results_count, dim=0)

    
    print(type(video_frames[0]))
    # Display the top 3 frames
    final_frames = []
    for frame_id in best_photo_idx:
      final_frames.append(video_frames[frame_id])
    return final_frames

      

  frames = search_video(query)
  return frames