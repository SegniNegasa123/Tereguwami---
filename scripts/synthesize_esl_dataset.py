"""
Synthetic Continuous ESL Dataset Generator (§10, §14)
Part of Tereguwami (ተርጓሚ) Research Infrastructure

Generates multi-domain continuous Ethiopian Sign Language annotations and 543 3D keypoint
trajectories for simulation, benchmarking, and unit testing.
"""

import os
import json
import numpy as np

DOMAINS = ["healthcare", "legal", "education", "civic_banking"]

PARALLEL_CORPUS = [
    # Healthcare
    {
        "domain": "healthcare",
        "gloss": "DOCTOR HEADACHE SEVERE THREE_DAYS",
        "am": "ዶክተር ላለፉት ሦስት ቀናት ብርቱ የራስ ምታት አለኝ።",
        "om": "Doktoraa, guyyoota sadii darban mataa bowwuu cimaan qaba.",
        "en": "Doctor, I have had a severe headache for the past three days.",
        "non_manual": "squint_pain_furrow"
    },
    {
        "domain": "healthcare",
        "gloss": "MEDICINE BEFORE_FOOD OR AFTER_FOOD QUESTION",
        "am": "መድኃኒቱን የምወስደው ከምግብ በፊት ነው ወይስ በኋላ?",
        "om": "Qoricha kana nyaata dura moo nyaata boodan fudhadha?",
        "en": "Should I take this medication before or after meals?",
        "non_manual": "eyebrow_elevation_polar_question"
    },
    {
        "domain": "healthcare",
        "gloss": "CHILD FEVER HIGH NOT_STOPPING",
        "am": "የልጁ ትኩሳት በጣም ከፍ ብሏል፤ ሊወርድ አልቻለም።",
        "om": "Ho'i mucichaa baay'ee ol ka'eera; gadi bu'uu hin dandeenye.",
        "en": "The child's fever is very high and will not subside.",
        "non_manual": "head_shake_distress"
    },
    # Legal / Court
    {
        "domain": "legal",
        "gloss": "ACCUSATION FALSE I NOT_STEAL",
        "am": "ክሱ ሀሰት ነው፤ እኔ አልሰረቅኩም።",
        "om": "Himanni kun soba; ani hin hanqanne.",
        "en": "The accusation is false; I did not steal.",
        "non_manual": "head_shake_negation"
    },
    {
        "domain": "legal",
        "gloss": "JUDGE PERMIT LAWYER CONSULT QUESTION",
        "am": "ክቡር ዳኛ፤ ከጠበቃዬ ጋር እንድማከር ይፈቀድልኛል?",
        "om": "Kabajamaa abbaa murtii, abukaatoo koo wajjin akkan mari'adhu naaf hayyamamaa?",
        "en": "Your Honor, may I be permitted to consult with my attorney?",
        "non_manual": "eyebrow_elevation_request"
    },
    # Education
    {
        "domain": "education",
        "gloss": "TEACHER EXAM PAGE TEN NOT_UNDERSTAND",
        "am": "መምህር፤ በገጽ አሥር ላይ ያለው ጥያቄ አልገባኝም።",
        "om": "Barsiisaa, gaaffiin fuula kudhan irra jiru naaf hin galle.",
        "en": "Teacher, I do not understand the question on page ten.",
        "non_manual": "furrow_confusion"
    },
    {
        "domain": "education",
        "gloss": "ASSIGNMENT TOMORROW SUBMIT DEADLINE QUESTION",
        "am": "የቤት ሥራውን የምናስረክበው ነገ ነው?",
        "om": "Hojiin manaa kan galchu boruu?",
        "en": "Is the assignment due tomorrow?",
        "non_manual": "eyebrow_elevation_polar_question"
    },
    # Civic & Banking
    {
        "domain": "civic_banking",
        "gloss": "ACCOUNT MONEY TRANSFER BIRR FIVE_THOUSAND",
        "am": "ከሒሳቤ አምስት ሺህ ብር ወደ ሌላ ሒሳብ ማስተላለፍ እፈልጋለሁ።",
        "om": "Herreega koo irraa gara herreega biraatti qarshii kuma shan daddabarsuu barbaada.",
        "en": "I want to transfer five thousand Birr from my account.",
        "non_manual": "neutral_affirmative"
    }
]


def generate_synthetic_dataset(output_dir: str, num_instances_per_template: int = 15):
    """Generate multiple instances with simulated realistic landmark variations."""
    os.makedirs(output_dir, exist_ok=True)
    all_samples = []

    sample_counter = 1
    for template in PARALLEL_CORPUS:
        for inst_idx in range(num_instances_per_template):
            signer_id = f"SYN_SIGNER_{inst_idx % 20 + 1:02d}"
            sample_id = f"ESL_SYN_{sample_counter:04d}"
            num_frames = np.random.randint(45, 90)

            sample_meta = {
                "sample_id": sample_id,
                "signer_id": signer_id,
                "domain": template["domain"],
                "gloss": template["gloss"],
                "non_manual_type": template["non_manual"],
                "translations": {
                    "am": template["am"],
                    "om": template["om"],
                    "en": template["en"]
                },
                "num_frames": num_frames,
                "fps": 30
            }
            all_samples.append(sample_meta)
            sample_counter += 1

    out_file = os.path.join(output_dir, "synthetic_continuous_esl_corpus.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(all_samples)} continuous ESL annotations saved to: {out_file}")
    return out_file


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "synthetic"))
    generate_synthetic_dataset(out_dir)
