import fitz  
import os

class DataExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text_from_pdf(self):
        """Extracts all text from the PDF."""
        text = ""
        try:
            with fitz.open(self.pdf_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF with fitz: {e}")
            return ""

    def extract_images_from_pdf(self, output_dir="output_images"):
        """
        Finds all image components on a page, groups nearby components into
        clusters.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pdf = fitz.open(self.pdf_path)
        image_paths = []
        total_image_count = 0
        
        # A margin to consider images "close" to each other (in pixels)
        proximity_margin = 40 

        for page_num, page in enumerate(pdf):
            # Get info for all image components on the page
            image_info_list = page.get_image_info(xrefs=True)
            if not image_info_list:
                continue

            # Get the bounding boxes for all image components
            rects = [fitz.Rect(info['bbox']) for info in image_info_list]
            
            # --- Clustering Logic ---
            clusters = []
            visited_indices = set()

            for i, rect1 in enumerate(rects):
                if i in visited_indices:
                    continue
                
                # Start a new cluster with the current rectangle
                current_cluster_indices = {i}
                # Use a queue for breadth-first search to find connected components
                queue = [i]
                visited_indices.add(i)

                while queue:
                    current_idx = queue.pop(0)
                    current_rect = rects[current_idx]
                    
                    # Check against all other rectangles
                    for j, rect2 in enumerate(rects):
                        if j not in visited_indices:
                            # If rect2 is close to current_rect, add it to the cluster
                            expanded_rect = current_rect + (-proximity_margin, -proximity_margin, proximity_margin, proximity_margin)
                            if expanded_rect.intersects(rect2):
                                visited_indices.add(j)
                                current_cluster_indices.add(j)
                                queue.append(j)
                clusters.append([rects[k] for k in current_cluster_indices])
            
            # --- Save a screenshot for each identified cluster ---
            for i, cluster in enumerate(clusters):
                total_image_count += 1
                
                # Calculate the total bounding box for the entire cluster
                total_bbox = fitz.Rect()
                for rect in cluster:
                    total_bbox.include_rect(rect)

                # Take a screenshot of the cluster's bounding box at high resolution
                pix = page.get_pixmap(clip=total_bbox, dpi=200)

                image_filename = os.path.join(output_dir, f"figure_{total_image_count}.png")
                pix.save(image_filename)
                image_paths.append(image_filename)

        print(f"Extracted {len(image_paths)} distinct figures.")
        return image_paths
