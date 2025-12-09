from pycocotools import mask as maskUtils
import logging

logging.basicConfig(level=logging.INFO)

def test_polygons():
    print("Testing polygons...")
    h, w = 100, 100
    # Provide a polygon as list of lists
    poly = [[10, 10, 50, 10, 50, 50, 10, 50]]
    
    try:
        # 1. frPyObjects
        rles = maskUtils.frPyObjects(poly, h, w)
        print(f"frPyObjects result type: {type(rles)}")
        if isinstance(rles, list):
            print(f"frPyObjects result[0] type: {type(rles[0])}")
            
        # 2. merge
        merged = maskUtils.merge(rles)
        print(f"merge result type: {type(merged)}")
        
        # 3. iou
        iscrowd = [0]
        # iou expects lists of RLEs
        ious = maskUtils.iou([merged], [merged], iscrowd)
        print(f"iou result: {ious}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

test_polygons()
