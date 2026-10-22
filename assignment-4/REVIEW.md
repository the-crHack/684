# Labels for the authored base/variant dataset

The CSV texts, IDs, labels, and explanations are preserved as supplied. This review
is a contextual-norm assessment, not independent human annotation or a legal ruling.
Difficulty remains an untested prediction until inference is measured.

## Cases requiring sensitivity analysis

| Variant | Authored label | Issue | Scoring policy |
|---|---:|---|---|
| 004a | 0 | The stated release exceeds the authorization's immunization-only scope. However, the recipient is a new treating physician, and HIPAA can permit treatment disclosures independently of authorization. “Counseling notes” does not establish that the records are separately maintained psychotherapy notes. A scoped-release institutional norm supports 0; an independent treatment basis could support 1. | Retain 0; also score alternative 1 and exclude from review-clear metrics. |
| 010b | 1 | Replacing direct identifiers and privacy-officer review do not alone establish de-identification. Discharge month and residual clinical attributes can matter; the text supplies neither Safe Harbor compliance nor an Expert Determination. An approved low-risk release could support 1, but identifiability remains unresolved. | Retain 1; also score alternative 0 and exclude from review-clear metrics. |

## Other requested checks

| Variant | Label | Assessment under the stated facts |
|---|---:|---|
| 002b | 1 | The verified patient requests personal unencrypted delivery after being informed of the risk. HHS permits patient email communication with reasonable safeguards; sensitive content is not itself a reason to deny the patient's access. Assumes the destination is the patient's chosen address and applicable safeguards are met. |
| 004b | 1 | The named recipient and sensitive record categories are expressly covered. This assumes ordinary clinical records and an otherwise valid authorization, not separately maintained psychotherapy notes or additional Part 2/state-law restrictions not established by the text. Initialing categories is not a universal substitute for every special consent requirement. |
| 006b | 1 | The narrative explicitly supplies marketing authorization covering name, email, condition and communication vendors, plus a one-send/delete restriction. This is a scoped newsletter service, rather than the base's ad-matching/lookalike reuse. Assumes a valid authorization and appropriate vendor agreement; agency involvement alone does not make it inappropriate. |
| 007a | 0 | Matching DOB/badge information establishes neither patient permission nor authority to receive a diagnosis. The employer's secondhand assertion does not establish the patient's wishes. |
| 007b | 1 | A signed release names the recipient and limits disclosure to absence dates; a callback verifies identity and no diagnosis is disclosed. |
| 003b / 005b | 1 | Sensitive diagnoses and medication/claim details can be relevant to assigned care or payment. A substance-use diagnosis alone does not establish that a particular record is subject to additional Part 2 disclosure restrictions. |

The remaining variants have clear role, purpose, audience or synthetic-data cues
consistent with their authored labels. For 008b, appropriateness relies on the
explicit “all test data, no real patients” premise; synthetic data derived from real
patients would require a different assessment.

Sources checked October 22, 2026:

- [HHS: treatment disclosures and psychotherapy notes](https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html)
- [HHS: email communication with patients](https://www.hhs.gov/hipaa/for-professionals/faq/does-hipaa-permit-health-care-providers-to-use-email-to-discuss-health-issues-with-patients/index.html)
- [HHS: marketing and business-associate communications](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/marketing/index.html)
- [HHS: Safe Harbor and Expert Determination](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html)
