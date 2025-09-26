"""
Advanced OCR Preprocessing Module

This module provides comprehensive image preprocessing for optimal Japanese OCR results.
Uses OpenCV and PIL for image enhancement, noise reduction, and text preparation.

Key improvements:
- Upscaling for better character recognition
- Bilateral filtering for noise reduction while preserving edges
- Adaptive thresholding for varying lighting conditions
- Morphological operations for text cleanup
- Japanese-specific optimizations
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def preprocess_image_for_ocr(image: Image.Image, 
                           scale_factor: float = 3.0,
                           apply_bilateral_filter: bool = True,
                           apply_morphological: bool = True,
                           enhance_contrast: bool = True) -> Image.Image:
    """
    Comprehensive image preprocessing for optimal Japanese OCR results.
    
    Args:
        image: Input PIL Image
        scale_factor: Image upscaling factor (1.0-5.0)
        apply_bilateral_filter: Apply bilateral filtering for noise reduction
        apply_morphological: Apply morphological operations for text cleanup
        enhance_contrast: Enhance image contrast and sharpness
    
    Returns:
        Preprocessed PIL Image ready for OCR
    """
    try:
        logger.debug(f"Starting OCR preprocessing with scale_factor={scale_factor}")
        
        # Convert PIL to OpenCV format
        cv_image = pil_to_cv2(image)
        
        # Step 1: Upscale image for better character recognition
        if scale_factor > 1.0:
            cv_image = upscale_image(cv_image, scale_factor)
            logger.debug(f"Upscaled image by factor {scale_factor}")
        
        # Step 2: Convert to grayscale if needed
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image.copy()
        
        # Step 3: Apply bilateral filter to reduce noise while preserving edges
        if apply_bilateral_filter:
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            logger.debug("Applied bilateral filtering")
        
        # Step 4: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            logger.debug("Applied CLAHE contrast enhancement")
        
        # Step 5: Adaptive thresholding for varying lighting conditions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        logger.debug("Applied adaptive thresholding")
        
        # Step 6: Morphological operations to clean up text
        if apply_morphological:
            binary = apply_morphological_operations(binary)
            logger.debug("Applied morphological operations")
        
        # Step 7: Convert back to PIL format
        result_image = cv2_to_pil(binary)
        
        # Step 8: Final PIL enhancements
        if enhance_contrast:
            result_image = enhance_pil_image(result_image)
            logger.debug("Applied PIL enhancements")
        
        logger.info(f"OCR preprocessing completed successfully. Final size: {result_image.size}")
        return result_image
        
    except Exception as e:
        logger.error(f"Error in OCR preprocessing: {e}")
        return image  # Return original image if preprocessing fails


def enhance_japanese_text(image: Image.Image) -> Image.Image:
    """
    Japanese-specific text enhancement optimizations.
    
    Args:
        image: Input PIL Image
    
    Returns:
        Enhanced PIL Image optimized for Japanese characters
    """
    try:
        logger.debug("Starting Japanese text enhancement")
        
        # Convert to OpenCV
        cv_image = pil_to_cv2(image)
        
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image.copy()
        
        # Japanese-specific preprocessing
        # 1. Gaussian blur to smooth small noise that affects Japanese characters
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. Morphological gradient to enhance character edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # 3. Combine original with gradient
        enhanced = cv2.addWeighted(gray, 0.7, morph_grad, 0.3, 0)
        
        # 4. Final adaptive threshold with parameters optimized for Japanese
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 3
        )
        
        # 5. Small morphological close to connect broken parts of characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        result = cv2_to_pil(binary)
        logger.debug("Japanese text enhancement completed")
        return result
        
    except Exception as e:
        logger.error(f"Error in Japanese text enhancement: {e}")
        return image


def upscale_image(cv_image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Upscale image using high-quality interpolation.
    
    Args:
        cv_image: OpenCV image array
        scale_factor: Scaling factor
    
    Returns:
        Upscaled OpenCV image
    """
    height, width = cv_image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Use INTER_CUBIC for better quality on upscaling
    upscaled = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return upscaled


def apply_morphological_operations(binary_image: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean up binary text image.
    
    Args:
        binary_image: Binary OpenCV image
    
    Returns:
        Cleaned binary image
    """
    # Remove small noise with opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
    # Fill gaps in characters with closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed


def enhance_pil_image(image: Image.Image) -> Image.Image:
    """
    Apply final PIL-based enhancements.
    
    Args:
        image: PIL Image
    
    Returns:
        Enhanced PIL Image
    """
    # Sharpen the image slightly
    enhancer = ImageEnhance.Sharpness(image)
    sharpened = enhancer.enhance(1.2)
    
    # Increase contrast slightly
    enhancer = ImageEnhance.Contrast(sharpened)
    contrasted = enhancer.enhance(1.1)
    
    return contrasted


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format."""
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    
    cv_image = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    if len(cv_image.shape) == 3:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    return cv_image


def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format."""
    if len(cv_image.shape) == 3:
        # Convert BGR to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(cv_image)
    return pil_image


def debug_save_image(image: Image.Image, 
                    filename: str, 
                    save_debug: bool = False) -> None:
    """
    Save image for debugging purposes.
    
    Args:
        image: PIL Image to save
        filename: Output filename
        save_debug: Whether to actually save (controlled by debug settings)
    """
    if save_debug:
        try:
            image.save(f"./debug_images/{filename}")
            logger.debug(f"Debug image saved: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save debug image: {e}")


def get_optimal_preprocessing_params(image_size: Tuple[int, int]) -> dict:
    """
    Get optimal preprocessing parameters based on image size.
    
    Args:
        image_size: Tuple of (width, height)
    
    Returns:
        Dictionary of optimal parameters
    """
    width, height = image_size
    total_pixels = width * height
    
    if total_pixels < 100000:  # Small image
        return {
            'scale_factor': 4.0,
            'apply_bilateral_filter': True,
            'apply_morphological': True,
            'enhance_contrast': True
        }
    elif total_pixels < 500000:  # Medium image
        return {
            'scale_factor': 3.0,
            'apply_bilateral_filter': True,
            'apply_morphological': True,
            'enhance_contrast': True
        }
    else:  # Large image
        return {
            'scale_factor': 2.0,
            'apply_bilateral_filter': False,  # Skip on large images for speed
            'apply_morphological': True,
            'enhance_contrast': True
        }


# Example usage and testing
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    # Test with a sample image (you would load your own)
    try:
        from PIL import Image
        
        # Create a test image with Japanese text (in practice, load from file)
        # test_image = Image.open("test_japanese.png")
        # processed = preprocess_image_for_ocr(test_image, scale_factor=3.0)
        # enhanced = enhance_japanese_text(processed)
        
        print("OCR preprocessing module loaded successfully")
        print("Available functions:")
        print("- preprocess_image_for_ocr()")
        print("- enhance_japanese_text()")
        print("- get_optimal_preprocessing_params()")
        
    except Exception as e:
        print(f"Error in module test: {e}")
