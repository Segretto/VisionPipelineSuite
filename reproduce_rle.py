from pycocotools import mask as maskUtils
import logging

logging.basicConfig(level=logging.INFO)

def test_uncompressed_rle():
    print("Testing uncompressed RLE (counts as list)...")
    h, w = 100, 100
    
    # Create simple RLE with counts as list
    # Uncompressed RLE: [h, w] and counts (list of run lengths)
    # Simple mask: 10 zeros, 10 ones, rest zeros.
    # Total pixels = 10000.
    # counts = [10, 10, 9980]
    
    rle_uncompressed = {
        "size": [h, w],
        "counts": [10, 10, 9980]
    }
    
    print(f"RLE uncompressed counts type: {type(rle_uncompressed['counts'])}") # <class 'list'>
    
    try:
        # Pass directly to iou
        # iou expects list of RLEs
        print("Attempting iou with uncompressed RLE...")
        ious = maskUtils.iou([rle_uncompressed], [rle_uncompressed], [0])
        print(f"iou result: {ious}")
    except Exception as e:
        print(f"FAILED (Direct): {e}")
        
    try:
        # Convert using frPyObjects
        print("Converting with frPyObjects...")
        # frPyObjects also handles list of RLEs? 
        # Signature: frPyObjects(pyObj, h, w)
        # If pyObj is RLE (dict), checks if it is uncompressed.
        # But frPyObjects expects list of objects usually?
        
        # Let's try passing list of 1 dict
        encoded = maskUtils.frPyObjects([rle_uncompressed], h, w)
        print(f"Encoded type: {type(encoded)}")
        print(f"Encoded[0] counts type: {type(encoded[0]['counts'])}") # Should be bytes
        
        ious = maskUtils.iou(encoded, encoded, [0])
        print(f"iou result (Converted): {ious}")
    except Exception as e:
        print(f"FAILED (Converted): {e}")
        import traceback
        traceback.print_exc()

test_uncompressed_rle()
