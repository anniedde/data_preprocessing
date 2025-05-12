import os
import os

for root, dirs, files in os.walk('/playpen-nas-ssd/awang/data/mystyle'):
    if 'test/anchors' in root or 'test/preprocessed' in root:
        for file in files:
            stem = file.split('.')[0]
            if stem.isdigit() and int(stem) > 9:
                os.remove(os.path.join(root, file))
                #rint(f'Bad file! {os.path.join(root, file)}')