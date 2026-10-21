# Scenario review

The root `assignment4_ci_scenarios.csv` is the untouched draft. The submission copy
is `data/scenarios.csv`; `revise_scenarios.py` reproduces the changes. All people and
records are fictional. Labels are normative judgments under the stated facts,
not independent human annotations or legal determinations.

| ID | Label | Review and distinct norm |
|---|---:|---|
| 001 | 1 | Assigned physician consultation for treatment. Clarified that relevant scope is institutional good practice, not HIPAA's minimum-necessary requirement for provider-to-provider treatment. |
| 002 | 1 | Approved aggregate vendor analysis. Added documented Expert Determination; suppression below 11 alone does not prove HIPAA de-identification. The approval is a synthetic scenario fact, not an assessment we performed on real data. |
| 003 | 1 | Patient-directed release to a named outside specialist, within a verified authorization. Authorization is sufficient here; it is not asserted to be necessary for every treatment disclosure. |
| 004 | 1 | Assigned nurse handoff for immediate safety. Replaced ambiguous bedside-to-station wording with private staff-room delivery. Unlike 001, this is a shift-transfer safety workflow rather than specialist consultation. |
| 005 | 1 | Scoped claim correction to payer staff for payment. No unrelated clinical records. |
| 006 | 0 | Clinical analytics data repurposed for advertising without authorization. Encryption would not repair the missing purpose and recipient authority. |
| 007 | 0 | Identifiable support transcript posted publicly for debugging. Replaced the malformed phone number with a reserved fictional number. |
| 008 | 0 | Diagnosis disclosed to employer without consent or an emergency. Explicitly identified the telephone disclosure as the reviewed flow; the internal call log is its record. |
| 009 | 0 | Recipient outside approved research protocol; pseudonyms leave indirect identification risks. Removed Iris Walker's name from the outgoing message, because the draft accidentally disclosed a direct identifier while claiming names were removed. The annotation now identifies subject P-0441. |
| 010 | 0 | Personal friend outside billing workflow receives identifiable invoice. Confidentiality applies even without a profit motive. |

MediInsight AI was renamed SynthHealth to match Assignments 1-3. Hospital-only
documents now name the partnership explicitly so the startup setting is visible
in the input. These partner workflows are within SynthHealth's operating setting;
they do not imply SynthHealth staff can access all clinical messages.

Automated validation checks column order, unique IDs, nonempty fields, word counts,
tuple syntax, labels, class counts, and exact CI-flow duplicates. Semantic review
above assesses distinct purposes, roles, and norms. It cannot establish inter-rater
agreement. A future benchmark should include independent raters, disputed cases,
and a held-out test set. This small, explicitly signposted set is a demonstration.

Sources checked October 21, 2026:

- [HHS minimum necessary](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/minimum-necessary-requirement/index.html)
- [HHS de-identification](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html)
- [HHS Privacy Rule summary](https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html)
