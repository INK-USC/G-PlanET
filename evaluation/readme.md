Our evaluation requirement followed `coco-caption` as the evaluation code base.

To evaluate, add the result path in line 171 of `split_eval_no_group.py` then run it,
the formulation of the result path should be
```
{"truth": "STEP 1: Turn right and go the the coffee table in front of the couch. | STEP 2: Close the laptop that is on the table. | STEP 3: Pick up the laptop on the table carry it to the right and to the lamp that is on your right against the wall. | STEP 4: Hold the laptop up and turn on the lamp in front of you. | END", "predict": "STEP 1: Turn around and go to the bed. | STEP 2: Close the laptop and pick it up.  | STEP 3: Turn right", "id": "trial_T20190907_231611_105301-0"}
{"truth": "STEP 1: Turn right and go the the coffee table in front of the couch. | STEP 2: Close the laptop that is on the table. | STEP 3: Pick up the laptop on the table carry it to the right and to the lamp that is on your right against the wall. | STEP 4: Hold the laptop up and turn on the lamp in front of you. | END", "predict": "STEP 1: Turn around and go to the bed. | STEP 2: Close the laptop and pick it up.  | STEP 3: Turn right", "id": "trial_T20190907_231611_105301-0"}
...
```
