"""
Multi-Reference Identity Aggregation (MRIA) - Quality Score Module
This module implements three quality metrics for reference face images:
1. Sharpness (Laplacian variance)
2. Pose (ArcFace angle / landmark deviation)
3. Illumination (histogram-based uniformity)
"""

import cv2
import numpy as np
import torch
from typing import Dict, List


def compute_sharpness_score(image: np.ndarray) -> float:
    """
    Compute sharpness score using Laplacian variance.
    Higher variance indicates sharper image.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
    
    Returns:
        Sharpness score (normalized to 0-1 range)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    # Normalize (empirical range: 0-1000, we clamp and normalize)
    score = min(variance / 500.0, 1.0)
    
    return float(score)


def compute_pose_score(face_info: Dict) -> float:
    """
    Compute pose score based on face angle and landmark quality.
    Frontal faces with clear landmarks get higher scores.
    
    Args:
        face_info: Dictionary containing face detection info with 'pose' or 'kps' keys
    
    Returns:
        Pose score (0-1, higher is better/more frontal)
    """
    # Method 1: If pose angles are available (yaw, pitch, roll)
    if 'pose' in face_info:
        pose = face_info['pose']
        # Compute deviation from frontal pose
        yaw = abs(pose[0]) if len(pose) > 0 else 0
        pitch = abs(pose[1]) if len(pose) > 1 else 0
        roll = abs(pose[2]) if len(pose) > 2 else 0
        
        # Penalize large angles (angles in degrees, typically -180 to 180)
        angle_deviation = (yaw + pitch + roll) / 3.0
        score = max(0, 1.0 - angle_deviation / 45.0)  # Good if angles < 45 degrees
        return float(score)
    
    # Method 2: If only landmarks available, check symmetry
    if 'kps' in face_info:
        kps = face_info['kps']  # Shape: (5, 2) for 5 landmarks
        
        # Check facial symmetry (left eye to nose vs right eye to nose)
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        
        # Compute distances
        left_dist = np.linalg.norm(left_eye - nose)
        right_dist = np.linalg.norm(right_eye - nose)
        
        # Symmetry score (closer to 1 means more symmetric/frontal)
        symmetry = min(left_dist, right_dist) / (max(left_dist, right_dist) + 1e-6)
        
        return float(symmetry)
    
    # Default score if no pose info available
    return 0.5


def compute_illumination_score(image: np.ndarray, bbox=None) -> float:
    """
    Compute illumination uniformity score using histogram analysis.
    Well-lit faces have more uniform intensity distribution.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        bbox: Optional bounding box [x1, y1, x2, y2] to crop face region
    
    Returns:
        Illumination score (0-1, higher means more uniform)
    """
    # Crop to face region if bbox provided
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        face_region = image[y1:y2, x1:x2]
    else:
        face_region = image
    
    # Convert to grayscale
    if len(face_region.shape) == 3:
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_region
    
    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    
    # Method 1: Entropy (higher entropy = more uniform)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    entropy_score = entropy / np.log(256)  # Normalize to 0-1
    
    # Method 2: Standard deviation (penalize high contrast/poor lighting)
    std_dev = gray.std()
    std_score = 1.0 - min(std_dev / 80.0, 1.0)  # Lower std is better
    
    # Method 3: Check for over/under exposure
    mean_intensity = gray.mean()
    exposure_score = 1.0 - abs(mean_intensity - 128) / 128.0  # Penalize too dark/bright
    
    # Combine scores (weighted average)
    illumination_score = 0.4 * entropy_score + 0.3 * exposure_score + 0.3 * std_score
    
    return float(illumination_score)


def compute_quality_scores(
    image: np.ndarray,
    face_info: Dict,
    weights: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Compute all quality scores for a reference face.
    
    Args:
        image: RGB image as numpy array
        face_info: Face detection info containing bbox, landmarks, etc.
        weights: Optional weights for each metric (default: equal weights)
    
    Returns:
        Dictionary with individual scores and overall quality score
    """
    if weights is None:
        weights = {
            'sharpness': 0.5,
            'pose': 0.3,
            'illumination': 0.2
        }
    
    # Compute individual scores
    s_sharp = compute_sharpness_score(image)
    s_pose = compute_pose_score(face_info)
    
    bbox = face_info.get('bbox', None)
    s_illum = compute_illumination_score(image, bbox)
    
    # Compute weighted overall score
    q_i = (
        weights['sharpness'] * s_sharp +
        weights['pose'] * s_pose +
        weights['illumination'] * s_illum
    )
    
    return {
        'sharpness': s_sharp,
        'pose': s_pose,
        'illumination': s_illum,
        'overall': q_i
    }


def aggregate_embeddings_with_quality(
    embeddings: List[torch.Tensor],
    quality_scores: List[float],
    method: str = 'weighted_average'
) -> torch.Tensor:
    """
    Aggregate multiple reference embeddings using quality-based weighting.
    
    Args:
        embeddings: List of embedding tensors (each shape: [1, D] or [D])
        quality_scores: List of quality scores for each embedding
        method: Aggregation method ('weighted_average', 'softmax', 'top_k')
    
    Returns:
        Aggregated embedding tensor
    """
    # Stack embeddings
    if embeddings[0].dim() == 1:
        emb_stack = torch.stack(embeddings, dim=0)  # [N, D]
    else:
        emb_stack = torch.cat(embeddings, dim=0)  # [N, D]
    
    # Convert quality scores to weights
    quality_tensor = torch.tensor(quality_scores, dtype=emb_stack.dtype, device=emb_stack.device)
    
    if method == 'weighted_average':
        # Simple weighted average
        weights = quality_tensor / quality_tensor.sum()
        weights = weights.unsqueeze(1)  # [N, 1]
        aggregated = (emb_stack * weights).sum(dim=0, keepdim=True)  # [1, D]
        
    elif method == 'softmax':
        # Softmax weighting (amplifies differences)
        temperature = 2.0
        weights = torch.softmax(quality_tensor / temperature, dim=0)
        weights = weights.unsqueeze(1)
        aggregated = (emb_stack * weights).sum(dim=0, keepdim=True)
        
    elif method == 'top_k':
        # Only use top-k highest quality embeddings
        k = min(3, len(quality_scores))
        top_k_indices = torch.topk(quality_tensor, k).indices
        selected_embs = emb_stack[top_k_indices]
        aggregated = selected_embs.mean(dim=0, keepdim=True)
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    # Normalize the aggregated embedding
    aggregated = aggregated / torch.norm(aggregated, dim=1, keepdim=True)
    
    return aggregated


if __name__ == "__main__":
    # Test the quality score functions
    print("Testing MRIA Quality Score Module...")
    
    # Test with a dummy image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_face_info = {
        'bbox': [100, 100, 400, 400],
        'kps': np.array([
            [200, 200],  # left eye
            [300, 200],  # right eye
            [250, 280],  # nose
            [220, 350],  # left mouth
            [280, 350],  # right mouth
        ])
    }
    
    scores = compute_quality_scores(test_image, test_face_info)
    print("Quality Scores:", scores)
    
    # Test aggregation
    dummy_embs = [torch.randn(1, 512) for _ in range(3)]
    dummy_scores = [0.7, 0.5, 0.9]
    
    aggregated = aggregate_embeddings_with_quality(dummy_embs, dummy_scores)
    print(f"Aggregated embedding shape: {aggregated.shape}")
    print("✓ MRIA module test passed!")

