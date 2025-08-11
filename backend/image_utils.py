import fitz
from pathlib import Path
from typing import List

def extract_and_cluster_images(pdf_path: str, output_dir: Path) -> List[str]:
    """
    Finds image components, groups them into clusters, and saves a single
    high-resolution screenshot for each cluster.
    """
    output_dir.mkdir(exist_ok=True)
    pdf = fitz.open(pdf_path)
    final_image_names = []
    total_image_count = 0
    proximity_margin = 40  # Margin to consider images "close"

    for page in pdf:
        image_info_list = page.get_image_info(xrefs=True)
        if not image_info_list:
            continue

        rects = [fitz.Rect(info['bbox']) for info in image_info_list]
        
        # Clustering Logic
        clusters = []
        visited_indices = set()
        for i, rect1 in enumerate(rects):
            if i in visited_indices:
                continue
            
            current_cluster_indices = {i}
            queue = [i]
            visited_indices.add(i)

            while queue:
                current_idx = queue.pop(0)
                current_rect = rects[current_idx]
                
                for j, rect2 in enumerate(rects):
                    if j not in visited_indices:
                        expanded_rect = current_rect + (-proximity_margin, -proximity_margin, proximity_margin, proximity_margin)
                        if expanded_rect.intersects(rect2):
                            visited_indices.add(j)
                            current_cluster_indices.add(j)
                            queue.append(j)
            clusters.append([rects[k] for k in current_cluster_indices])
        
        # Save a screenshot for each cluster
        for cluster in clusters:
            total_image_count += 1
            total_bbox = fitz.Rect()
            for rect in cluster:
                total_bbox.include_rect(rect)

            pix = page.get_pixmap(clip=total_bbox, dpi=200)
            image_filename = f"figure_{total_image_count}.png"
            pix.save(output_dir / image_filename)
            final_image_names.append(image_filename)

    pdf.close()
    return final_image_names