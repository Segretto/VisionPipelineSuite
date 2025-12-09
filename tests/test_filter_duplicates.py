import unittest
import copy
from vision_suite.utils.coco import filter_coco_duplicates

class TestFilterDuplicates(unittest.TestCase):
    def setUp(self):
        # Create a dummy COCO dataset
        self.coco_data = {
            "images": [
                {"id": 1, "width": 100, "height": 100, "file_name": "test.jpg"}
            ],
            "categories": [
                {"id": 1, "name": "cat"}
            ],
            "annotations": [
                # Ann 1: Base square [10, 10, 50, 50] (area 1600)
                {
                    "id": 1, "image_id": 1, "category_id": 1,
                    "bbox": [10, 10, 40, 40], # x, y, w, h
                    "area": 1600,
                    "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
                    "iscrowd": 0
                },
                # Ann 2: Identical to Ann 1 (IoU 1.0) - should be removed (smaller ID usually kept? Logic removes smaller area. If equal area? Logic remove 'j' if area_i < area_j else remove 'j'... wait logic: if area_i < area_j remove i, else remove j. If equal area, j is removed.)
                {
                    "id": 2, "image_id": 1, "category_id": 1,
                    "bbox": [10, 10, 40, 40],
                    "area": 1600,
                    "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
                    "iscrowd": 0
                },
                # Ann 3: Slightly shifted, high overlap (IoU > 0.95)
                # Box [11, 11, 40, 40]
                {
                    "id": 3, "image_id": 1, "category_id": 1,
                    "bbox": [11, 11, 40, 40],
                    "area": 1600, # Same area
                    "segmentation": [[11, 11, 51, 11, 51, 51, 11, 51]],
                    "iscrowd": 0
                },
                # Ann 4: Distinct object (IoU < 0.95)
                {
                    "id": 4, "image_id": 1, "category_id": 1,
                    "bbox": [60, 60, 20, 20],
                    "area": 400,
                    "segmentation": [[60, 60, 80, 60, 80, 80, 60, 80]],
                    "iscrowd": 0
                }
            ]
        }

    def test_exact_duplicate_removal(self):
        # Test just 1 and 2
        data = copy.deepcopy(self.coco_data)
        data["annotations"] = [self.coco_data["annotations"][0], self.coco_data["annotations"][1]]
        
        filtered = filter_coco_duplicates(data, iou_threshold=0.95)
        # Should have 1 annotation left
        self.assertEqual(len(filtered["annotations"]), 1)
        # ID 1 should be kept because loop: i=0 (id1), j=1 (id2). area1=1600, area2=1600. area1 < area2 is False. Else remove j (id2).
        self.assertEqual(filtered["annotations"][0]["id"], 1)

    def test_high_overlap_removal(self):
        # Test 1 and 3. IoU should be high ~0.9. Let's make sure it's > 0.95
        # 40x40 box. Intersection 39x39. IoU = 39*39 / (2 * 40*40 - 39*39) = 1521 / (3200 - 1521) = 1521 / 1679 = 0.90
        # Wait, my manual calculation for 1px shift IoU might be < 0.95.
        # IoU for [10,50] vs [11,51]: Intersection is [11,50] (39x39). Union is [10,51] (41x41) - 2 corners... 
        # roughly 0.9.
        # Let's adjust Ann 3 to be very very close.
        # Just use identical polygon for the "high overlap" test to be safe, or make Ann 3 contained in Ann 1 but smaller.
        
        # Scenario: Nested small box inside big box.
        # Big: 100x100. Small: 99x99 inside. IoU ~ 0.98.
        # Smaller (99x99) should be removed.
        
        data = copy.deepcopy(self.coco_data)
        ann_big = {
             "id": 10, "image_id": 1, "category_id": 1,
             "bbox": [0, 0, 100, 100], "area": 10000,
             "segmentation": [[0,0, 100,0, 100,100, 0,100]],
             "iscrowd": 0
        }
        ann_small = {
             "id": 11, "image_id": 1, "category_id": 1,
             "bbox": [0, 0, 99, 99], "area": 9801,
             "segmentation": [[0,0, 99,0, 99,99, 0,99]],
             "iscrowd": 0
        }
        data["annotations"] = [ann_big, ann_small]
        
        filtered = filter_coco_duplicates(data, iou_threshold=0.95)
        self.assertEqual(len(filtered["annotations"]), 1)
        self.assertEqual(filtered["annotations"][0]["id"], 10) # Big one kept

    def test_distinct_objects(self):
        # Test 1 and 4
        data = copy.deepcopy(self.coco_data)
        data["annotations"] = [self.coco_data["annotations"][0], self.coco_data["annotations"][3]]
        
        filtered = filter_coco_duplicates(data, iou_threshold=0.95)
        self.assertEqual(len(filtered["annotations"]), 2)

    def test_bbox_only(self):
        # Test without segmentation keys
        data = copy.deepcopy(self.coco_data)
        ann1 = data["annotations"][0]
        del ann1["segmentation"]
        ann2 = data["annotations"][1]
        del ann2["segmentation"]
        
        data["annotations"] = [ann1, ann2]
        
        filtered = filter_coco_duplicates(data, iou_threshold=0.95)
        self.assertEqual(len(filtered["annotations"]), 1)

if __name__ == "__main__":
    unittest.main()
