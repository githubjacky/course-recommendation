from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
from pathlib import Path
import json

def main():
    course_feature = json.loads(
        Path('data/model_data/course_feature.json').read_text()
    )
    doc = list(course_feature.values())
        ws_driver = CkipWordSegmenter(model="bert-base", device=0)
        pos_driver = CkipPosTagger(model="bert-base", device=0)
        ws_res = ws_driver(doc)
        pos_res = pos_driver(ws_res)
        clean_doc = self.clean(ws_res, pos_res)
        course_feature = {
            str(idx): i
            for idx, i in enumerate(clean_doc)
        }



if __name__ == '__main__':
    main()