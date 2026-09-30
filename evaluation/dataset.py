from typing import List, Dict, Any
from pydantic import BaseModel, Field

class EvaluationSample(BaseModel):
    """
    Curated evaluation instance containing a realistic clinical question,
    expected relevant chunk IDs, and reference clinical ground truth.
    """
    sample_id: str
    query: str
    topic: str
    relevant_chunk_ids: List[str]
    ground_truth_answer: str
    clinical_context: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

def get_curated_evaluation_dataset() -> List[EvaluationSample]:
    """
    Returns a benchmark dataset of 6 realistic physiotherapy questions across
    orthopaedic, sports, and neurological clinical rehabilitation domains.
    """
    return [
        EvaluationSample(
            sample_id="eval_acl_rts",
            query="What quadriceps strength symmetry index is recommended before return to unrestricted sports after ACL reconstruction?",
            topic="ACL Rehabilitation",
            relevant_chunk_ids=["acl_chunk_01", "acl_chunk_02"],
            ground_truth_answer="A limb symmetry index (LSI) of at least 90% in quadriceps strength and hop testing is recommended before clearance for unrestricted sports.",
            clinical_context="Post-operative ACL rehabilitation return-to-sport criteria."
        ),
        EvaluationSample(
            sample_id="eval_pfps_hip",
            query="Does proximal hip strengthening improve pain and function in patellofemoral pain syndrome compared to knee exercises alone?",
            topic="Patellofemoral Pain",
            relevant_chunk_ids=["pfps_chunk_01"],
            ground_truth_answer="Combining posterolateral hip muscle strengthening (abductors and external rotators) with knee-focused exercises produces superior pain relief and functional improvement compared to knee exercises alone.",
            clinical_context="Patellofemoral pain clinical management."
        ),
        EvaluationSample(
            sample_id="eval_achilles_hsr",
            query="How does heavy slow resistance training compare to the Alfredson eccentric protocol for midportion Achilles tendinopathy?",
            topic="Achilles Tendinopathy",
            relevant_chunk_ids=["achilles_chunk_01", "achilles_chunk_02"],
            ground_truth_answer="Heavy slow resistance (HSR) training demonstrates equivalent clinical outcomes (VISA-A scores) to the Alfredson eccentric protocol at 12 weeks, with higher patient satisfaction and lower time burden.",
            clinical_context="Tendon loading protocols."
        ),
        EvaluationSample(
            sample_id="eval_lbp_mce",
            query="What is the evidence for motor control exercise versus general aerobic exercise in chronic non-specific low back pain?",
            topic="Low Back Pain",
            relevant_chunk_ids=["lbp_chunk_01"],
            ground_truth_answer="Motor control exercises targeting transversus abdominis and multifidus show modest short-term improvements in pain, but produce outcomes comparable to general exercise and walking at long-term follow-up.",
            clinical_context="Lumbar spine rehabilitation."
        ),
        EvaluationSample(
            sample_id="eval_rotator_cuff_iso",
            query="Can isometric contractions provide acute analgesic pain relief in reactive rotator cuff tendinopathy?",
            topic="Rotator Cuff",
            relevant_chunk_ids=["rc_chunk_01"],
            ground_truth_answer="Heavy sustained isometric contractions (e.g., 5 repetitions of 45-second holds at 70% MVC) can produce an acute analgesic effect lasting up to 45 minutes in reactive tendinopathy.",
            clinical_context="Shoulder pain modulation."
        ),
        EvaluationSample(
            sample_id="eval_stroke_cimt",
            query="What is the minimum recommended daily constraint time for modified Constraint-Induced Movement Therapy (mCIMT) in subacute stroke?",
            topic="Neurorehabilitation",
            relevant_chunk_ids=["stroke_chunk_01"],
            ground_truth_answer="Modified CIMT typically prescribes wearing a mitt on the unaffected arm for 5 hours per day accompanied by 2 hours of structured task-specific practice 5 days per week over 2-3 weeks.",
            clinical_context="Stroke upper extremity motor recovery."
        )
    ]

def get_benchmark_corpus_chunks() -> List[Dict[str, Any]]:
    """
    Returns the benchmark corpus corresponding to the evaluation dataset.
    Can be loaded directly into any VectorStore for running reproducible evaluations.
    """
    return [
        {
            "chunk_id": "acl_chunk_01",
            "document_id": "doc_acl",
            "content": "A quadriceps limb symmetry index (LSI) greater than 90% is widely recognized as a critical criterion prior to return to unrestricted competitive pivoting sports after ACL reconstruction.",
            "section": "Results",
            "title": "Return to Sport Criteria Following Anterior Cruciate Ligament Reconstruction",
            "embedding": [0.85, 0.15, 0.05, 0.0]
        },
        {
            "chunk_id": "acl_chunk_02",
            "document_id": "doc_acl",
            "content": "Failing to achieve 90% quadriceps index and symmetrical hop test distances was associated with a 4-fold increase in secondary ACL reinjury rates.",
            "section": "Discussion",
            "title": "Return to Sport Criteria Following Anterior Cruciate Ligament Reconstruction",
            "embedding": [0.80, 0.20, 0.05, 0.0]
        },
        {
            "chunk_id": "pfps_chunk_01",
            "document_id": "doc_pfps",
            "content": "Hip abductor and external rotator strengthening in combination with quadriceps exercises resulted in significantly greater reduction in Anterior Knee Pain Scale (AKPS) scores than quadriceps exercise alone.",
            "section": "Results",
            "title": "Proximal Hip Versus Knee Focused Rehabilitation for Patellofemoral Pain",
            "embedding": [0.10, 0.85, 0.10, 0.0]
        },
        {
            "chunk_id": "achilles_chunk_01",
            "document_id": "doc_achilles",
            "content": "Heavy slow resistance (HSR) training using 3 sets of 6-8 repetitions at 80% 1RM yielded equivalent improvements in VISA-A scores compared to the Alfredson eccentric protocol at 12-week and 52-week follow-up.",
            "section": "Results",
            "title": "Heavy Slow Resistance Versus Eccentric Training in Achilles Tendinopathy",
            "embedding": [0.05, 0.15, 0.85, 0.0]
        },
        {
            "chunk_id": "achilles_chunk_02",
            "document_id": "doc_achilles",
            "content": "Patient compliance was significantly higher in the heavy slow resistance group (92%) compared to the Alfredson protocol (78%), owing to less frequent weekly sessions.",
            "section": "Discussion",
            "title": "Heavy Slow Resistance Versus Eccentric Training in Achilles Tendinopathy",
            "embedding": [0.05, 0.20, 0.80, 0.0]
        },
        {
            "chunk_id": "lbp_chunk_01",
            "document_id": "doc_lbp",
            "content": "Motor control exercise targeting deep stabilizing musculature produced clinically meaningful short-term pain relief, but demonstrated similar efficacy to general aerobic walking programs at 12 months.",
            "section": "Results",
            "title": "Motor Control Exercise Versus General Exercise for Chronic Low Back Pain",
            "embedding": [0.10, 0.10, 0.10, 0.85]
        },
        {
            "chunk_id": "rc_chunk_01",
            "document_id": "doc_rc",
            "content": "Five repetitions of 45-second isometric shoulder external rotation holds at 70% maximum voluntary contraction induced immediate pain reduction (average 2.8 points on NRS) lasting over 45 minutes.",
            "section": "Results",
            "title": "Isometric Exercise for Acute Pain Relief in Reactive Shoulder Tendinopathy",
            "embedding": [0.30, 0.10, 0.70, 0.10]
        },
        {
            "chunk_id": "stroke_chunk_01",
            "document_id": "doc_stroke",
            "content": "Modified constraint-induced movement therapy involving 5 hours of mitt restriction daily with 2 hours of functional task practice produced substantial gains on the Action Research Arm Test.",
            "section": "Results",
            "title": "Modified CIMT Protocols in Subacute Stroke Upper Limb Recovery",
            "embedding": [0.05, 0.05, 0.05, 0.90]
        },
        {
            "chunk_id": "distractor_chunk_01",
            "document_id": "doc_distractor",
            "content": "Dietary protein supplementation supports muscle protein synthesis during intense strength conditioning phases in elite endurance runners.",
            "section": "Discussion",
            "title": "Nutritional Strategies in Sports Performance",
            "embedding": [0.25, 0.25, 0.25, 0.25]
        }
    ]
