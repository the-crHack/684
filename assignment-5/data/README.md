# Data

`hospital_dev.jsonl` contains ten fictional development notes; `hospital_eval.jsonl` contains fifty evaluation notes. The notebook generates them deterministically with seed 684. Each record includes its split, layout, case ID, note, and expected clinic, booking contact, and interval. Context lines contain distracting contacts and numbers. All email domains are reserved `.example` domains; these are not patient records or clinical recommendations.

Layouts are shared across splits, while record IDs are disjoint. The overwrite guard prevents silent changes to an existing dataset.

`jbb_harmful.csv` is the harmful split of JailbreakBench/JBB-Behaviors, revision `886acc352a31533ffbcf4ef22c744658688086fc`. Its license is in `JBB_LICENSE.txt`. Behaviors 61, 64, and 69, which involve the sexual exploitation of minors, are excluded before screening. `jbb_benign.csv` is the benign split from the same revision, used to check the jailbreak defense for added refusals. Dataset source: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors.
