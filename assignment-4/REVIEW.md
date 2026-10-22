# Review of the revised scenarios

Inputs: `data/taskA_scenarios_10.csv` and `data/taskB_variants_20.csv`.
The texts, IDs, labels and reasons are preserved as supplied. This review evaluates
contextual norms under the stated facts; it is not independent human annotation.

The ten base documents are distinct, 116–144 words long, and balanced 5/5. The
20 variants are balanced 10/10, with one label-preserving and one label-flipping
variant per base. All joins agree with the revised base labels. `base_difficulty`
is a prediction (3 easy, 4 medium, 3 hard bases), never model input or an observed
accuracy claim.

## Changes resolving the previous ambiguities

| Case | Label | Assessment |
|---|---:|---|
| 004a | 0 | The recipient is now employer occupational health rather than a new treating physician. The authorization covers immunizations only, while the packet includes unrelated clinical and counseling records. The independent treatment-disclosure ambiguity is removed. |
| 010b | 1 | A signed story release and approval of the exact wording now authorize the identifiable publicity. The label no longer depends on uncertain de-identification. |

No variants are excluded from the revised experiment. The original flags and
results remain in `results/archive/tinker_original/` for the original dataset only.

## Assumptions worth retaining

| Case | Label | Assessment under the stated facts |
|---|---:|---|
| 002a | 0 | The spouse's own identity verification does not establish patient permission or representative authority. No patient agreement or care-related basis for this disclosure is supplied. |
| 002b | 1 | The verified patient requests unencrypted delivery to his chosen address after a risk warning. Sensitive content alone does not invalidate access; reasonable safeguards still apply. |
| 003b | 1 | The text supplies marketing authorization for name, email and condition, and a vendor restricted to the authorized send and deletion. Assumes an otherwise valid authorization and appropriate vendor arrangement. |
| 004b | 1 | The named recipient and sensitive categories are explicitly covered. Assumes valid applicable consents, ordinary mental health treatment records rather than separately maintained psychotherapy notes, and no unstated legal restrictions. Initialing categories alone is not a universal legal consent standard. |
| 005a | 0 | Unrelated complete-chart and counseling content exceeds a duplicate-claim correction's payment purpose. The label does not require treating every counseling note as a separately maintained psychotherapy note. |
| 005b | 1 | Diagnosis and identifiers can be relevant to a claim. An alcohol-related diagnosis alone does not establish that the record comes from a program subject to additional Part 2 rules. |
| 006b | 1 | Both nurses are assigned to the patient, and the exchange supports handoff and medication reconciliation. Informal language alone does not make it gossip. |
| 007a / 007b | 0 / 1 | A caller's secondhand claim and matching identifiers do not establish permission. A signed scoped release plus verified callback supports disclosure of absence dates only. |
| 008 / 008a | 0 | The BAA explicitly restricts use to Riverside scoring. Internal access, encryption, hashing names while retaining MRNs, and informal vendor approval do not authorize commercial model training. The base already grants access and copies PHI outside its agreed purpose, independently of the future termination date. |
| 008b | 1 | The narrative explicitly states completed Safe Harbor review, removal of identifiers and free text, privacy-office sign-off, and an amended agreement allowing product development. Accept those as scenario facts; hashing alone would not suffice. |
| 009a | 0 | The intended contrast is an override without an established treatment encounter or substantiated emergency, plus access to unrelated sensitive notes. Lack of registration alone is not proof of improper access: urgent treatment can precede registration. Interpret the label using the complete narrative, not that administrative field alone. |
| 009b | 1 | An incapacitated patient's bona fide emergency and necessary methadone dosing information support emergency disclosure. Assumes required Part 2 documentation is completed under the stated policy; an emergency override is not blanket permission to browse unrelated sensitive records. |
| 010 / 010a | 0 | The combination of occupation, town, dates, injury and confirming public comment identifies the patient. Staff approval and omission of a name or photo do not replace patient authorization. |

Sources checked October 22, 2026:

- [HHS: Privacy Rule, treatment and psychotherapy notes](https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html)
- [HHS: patient email](https://www.hhs.gov/hipaa/for-professionals/faq/does-hipaa-permit-health-care-providers-to-use-email-to-discuss-health-issues-with-patients/index.html)
- [HHS: marketing](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/marketing/index.html)
- [HHS: business associates](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/business-associates/index.html)
- [HHS: de-identification](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html)
- [42 CFR 2.51: medical emergencies and documentation](https://www.ecfr.gov/current/title-42/chapter-I/subchapter-A/part-2/subpart-D/section-2.51)
