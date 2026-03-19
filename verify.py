# verify.py - run this first!
import os

pairs = [
    ('dataset/images/train', 'dataset/labels/train'),
    ('dataset/images/val',   'dataset/labels/val'),
]

for img_dir, lbl_dir in pairs:
    imgs = set(f.replace('.jpg','').replace('.png','') 
               for f in os.listdir(img_dir))
    lbls = set(f.replace('.txt','') 
               for f in os.listdir(lbl_dir))
    
    missing = imgs - lbls
    print(f"\n{img_dir}")
    print(f"  Images : {len(imgs)}")
    print(f"  Labels : {len(lbls)}")
    print(f"  Missing labels: {len(missing)}")
    
    if len(missing) == 0:
        print("  Status: READY to train!")
    else:
        print(f"  Warning: {len(missing)} images have no label file")