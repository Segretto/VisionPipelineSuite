import json
import os
import sys
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_suite.utils.coco import validate_coco_json, merge_coco_jsons

def create_dummy_coco(path, images, annotations, categories):
    data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(path, 'w') as f:
        json.dump(data, f)

def test_validation():
    print("Testing Validation...")
    
    # Valid COCO
    valid_data = {
        "images": [{"id": 1, "file_name": "img1.jpg", "height": 100, "width": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]}],
        "categories": [{"id": 1, "name": "cat"}]
    }
    create_dummy_coco("valid.json", valid_data["images"], valid_data["annotations"], valid_data["categories"])
    
    res = validate_coco_json("valid.json")
    if res["valid"]:
        print("PASS: Valid JSON validated correctly.")
    else:
        print(f"FAIL: Valid JSON failed validation: {res['errors']}")

    # Invalid COCO (missing keys)
    with open("invalid.json", 'w') as f:
        json.dump({"foo": "bar"}, f)
    
    res = validate_coco_json("invalid.json")
    if not res["valid"] and "Missing required key" in res["errors"][0]:
        print("PASS: Invalid JSON detected correctly.")
    else:
        print(f"FAIL: Invalid JSON not detected correctly: {res}")

    # Invalid COCO (broken reference)
    broken_data = {
        "images": [{"id": 1, "file_name": "img1.jpg"}],
        "annotations": [{"id": 1, "image_id": 99, "category_id": 1}], # image 99 missing
        "categories": [{"id": 1, "name": "cat"}]
    }
    create_dummy_coco("broken.json", broken_data["images"], broken_data["annotations"], broken_data["categories"])
    
    res = validate_coco_json("broken.json")
    if not res["valid"] and "references non-existent image_id" in res["errors"][0]:
        print("PASS: Broken reference detected correctly.")
    else:
        print(f"FAIL: Broken reference not detected correctly: {res}")

    # Cleanup
    os.remove("valid.json")
    os.remove("invalid.json")
    os.remove("broken.json")

def test_merge():
    print("\nTesting Merge...")
    
    # Setup 2 JSONs with some overlap
    # Common image: img1.jpg
    # Unique images: img2.jpg (in A), img3.jpg (in B)
    
    cat_dog = {"id": 1, "name": "dog"}
    cat_cat = {"id": 2, "name": "cat"}
    
    # Dataset A
    imgs_a = [
        {"id": 1, "file_name": "img1.jpg"},
        {"id": 2, "file_name": "img2.jpg"}
    ]
    anns_a = [
        {"id": 1, "image_id": 1, "category_id": 1}, # dog on img1
        {"id": 2, "image_id": 2, "category_id": 1}  # dog on img2
    ]
    cats_a = [cat_dog]
    
    # Dataset B
    imgs_b = [
        {"id": 10, "file_name": "img1.jpg"}, # ID different, filename same
        {"id": 11, "file_name": "img3.jpg"}
    ]
    anns_b = [
        {"id": 10, "image_id": 10, "category_id": 5}, # cat (id 5) on img1
        {"id": 11, "image_id": 11, "category_id": 5}
    ]
    cats_b = [{"id": 5, "name": "cat"}] # name matches cat_cat, but ID is different
    
    create_dummy_coco("data_a.json", imgs_a, anns_a, cats_a)
    create_dummy_coco("data_b.json", imgs_b, anns_b, cats_b)
    
    merge_coco_jsons(["data_a.json", "data_b.json"], "merged.json")
    
    with open("merged.json", 'r') as f:
        merged = json.load(f)
        
    # Checks
    # 1. Should only have img1.jpg
    if len(merged["images"]) == 1 and merged["images"][0]["file_name"] == "img1.jpg":
        print("PASS: Correctly intersected images.")
    else:
        print(f"FAIL: Incorrect images in merged: {merged['images']}")
        
    # 2. Should have 2 annotations (dog and cat on img1)
    if len(merged["annotations"]) == 2:
        print("PASS: Correct number of annotations.")
    else:
        print(f"FAIL: Incorrect number of annotations: {len(merged['annotations'])}")
        
    # 3. Should have 2 categories (dog and cat)
    if len(merged["categories"]) == 2:
        names = {c["name"] for c in merged["categories"]}
        if "dog" in names and "cat" in names:
            print("PASS: Categories merged correctly.")
        else:
            print(f"FAIL: Categories mismatch: {names}")
            
    # Cleanup
    os.remove("data_a.json")
    os.remove("data_b.json")
    os.remove("merged.json")

if __name__ == "__main__":
    test_validation()
    test_merge()
