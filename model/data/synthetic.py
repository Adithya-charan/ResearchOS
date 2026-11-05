"""Deterministic synthetic research-style corpus.

This is *template-generated* text written for this project (licence: CC0-1.0). It exists so the
tokenizer/training/inference pipeline can be validated end to end with zero licensing
questions. A model trained on it learns the document structure and vocabulary of research
writing - it learns no real science. Real corpora go through the same manifest pipeline.
"""
from __future__ import annotations

import random
from collections.abc import Iterator

DOMAINS: dict[str, dict[str, list[str]]] = {
    "materials science": {
        "subjects": ["solid-state electrolytes", "perovskite thin films", "graphene composites", "lithium-sulfur cathodes", "metal-organic frameworks", "shape-memory alloys", "polymer membranes"],
        "properties": ["ionic conductivity", "thermal stability", "tensile strength", "charge mobility", "degradation rate", "surface roughness", "porosity"],
        "methods": ["X-ray diffraction", "impedance spectroscopy", "scanning electron microscopy", "density functional theory", "thermogravimetric analysis"],
        "units": ["mS/cm", "MPa", "percent", "cm2/Vs", "nm", "kelvin"],
    },
    "genomics": {
        "subjects": ["regulatory variants", "gene expression programs", "transposable elements", "methylation patterns", "splice junctions", "enhancer regions", "structural variants"],
        "properties": ["expression level", "allele frequency", "read coverage", "mutation rate", "methylation fraction", "splicing efficiency"],
        "methods": ["RNA sequencing", "chromatin immunoprecipitation", "whole-genome sequencing", "single-cell profiling", "differential expression analysis"],
        "units": ["transcripts per million", "percent", "reads per base", "fold change", "events per megabase"],
    },
    "climate science": {
        "subjects": ["Arctic sea ice", "monsoon rainfall", "ocean heat content", "soil carbon stocks", "glacier mass balance", "wildfire frequency", "cloud albedo"],
        "properties": ["annual extent", "seasonal variability", "warming trend", "carbon flux", "melt rate", "precipitation anomaly"],
        "methods": ["satellite altimetry", "ensemble climate simulation", "ice core analysis", "reanalysis products", "eddy covariance towers"],
        "units": ["million square kilometres", "millimetres per day", "gigatonnes", "watts per square metre", "percent"],
    },
    "neuroscience": {
        "subjects": ["hippocampal place cells", "cortical oscillations", "synaptic plasticity", "sleep spindles", "dopaminergic signalling", "visual cortex responses"],
        "properties": ["firing rate", "spike timing precision", "phase coupling", "learning rate", "response latency", "population synchrony"],
        "methods": ["two-photon imaging", "extracellular recording", "optogenetic stimulation", "functional magnetic resonance imaging", "computational modelling"],
        "units": ["hertz", "milliseconds", "percent", "standard deviations", "spikes per second"],
    },
    "machine learning": {
        "subjects": ["sequence models", "graph neural networks", "convolutional classifiers", "reinforcement learning agents", "sparse autoencoders", "retrieval systems"],
        "properties": ["validation accuracy", "sample efficiency", "calibration error", "training stability", "inference latency", "robustness to noise"],
        "methods": ["ablation study", "cross-validation", "hyperparameter sweep", "bootstrap resampling", "learning-curve analysis"],
        "units": ["percent", "milliseconds", "bits per token", "points", "epochs"],
    },
    "epidemiology": {
        "subjects": ["seasonal influenza", "vector-borne infections", "hospital-acquired infections", "vaccine uptake", "antimicrobial resistance", "waterborne outbreaks"],
        "properties": ["incidence rate", "reproduction number", "case fatality", "vaccination coverage", "time to diagnosis", "transmission probability"],
        "methods": ["cohort analysis", "case-control design", "compartmental modelling", "serological survey", "spatial clustering analysis"],
        "units": ["cases per 100000", "percent", "days", "ratio", "odds"],
    },
    "astronomy": {
        "subjects": ["exoplanet atmospheres", "supernova remnants", "galaxy clusters", "pulsar timing", "stellar streams", "protoplanetary discs"],
        "properties": ["luminosity", "orbital period", "metallicity", "redshift", "spectral index", "dust temperature"],
        "methods": ["transit photometry", "radial velocity monitoring", "spectroscopic survey", "radio interferometry", "N-body simulation"],
        "units": ["solar luminosities", "days", "dex", "kelvin", "milliarcseconds"],
    },
    "agronomy": {
        "subjects": ["wheat cultivars", "rice paddies", "legume intercrops", "orchard soils", "drought-tolerant maize", "cover crops"],
        "properties": ["grain yield", "nitrogen uptake", "water use efficiency", "root depth", "pest incidence", "soil organic matter"],
        "methods": ["randomised field trial", "lysimeter measurement", "remote sensing", "soil core sampling", "growth-chamber experiment"],
        "units": ["tonnes per hectare", "percent", "kilograms per hectare", "centimetres", "millimetres"],
    },
    "robotics": {
        "subjects": ["legged locomotion", "soft grippers", "visual odometry", "swarm coordination", "manipulation planners", "underwater vehicles"],
        "properties": ["tracking error", "energy consumption", "grasp success rate", "planning time", "localisation drift", "payload capacity"],
        "methods": ["hardware-in-the-loop testing", "motion capture evaluation", "simulation benchmark", "field deployment", "system identification"],
        "units": ["centimetres", "joules per metre", "percent", "milliseconds", "newtons"],
    },
    "chemistry": {
        "subjects": ["heterogeneous catalysts", "organic photovoltaics", "enzyme mimics", "ionic liquids", "peptide macrocycles", "zeolite frameworks"],
        "properties": ["turnover frequency", "selectivity", "binding affinity", "reaction yield", "solubility", "activation energy"],
        "methods": ["nuclear magnetic resonance", "mass spectrometry", "kinetic profiling", "calorimetry", "computational screening"],
        "units": ["per second", "percent", "kilojoules per mole", "millimolar", "grams per litre"],
    },
}

FACTORS = ["temperature", "dosage", "training duration", "sampling depth", "humidity", "pressure", "batch size", "exposure time", "annealing rate", "sample age", "ambient noise", "preprocessing choice"]
DESIGNS = ["controlled", "longitudinal", "comparative", "replication", "cross-sectional", "simulation-based", "multi-site", "pre-registered"]
HEDGES = ["modest", "statistically uncertain", "larger than expected", "consistent across replicates", "sensitive to the preprocessing pipeline", "smaller than reported in earlier work", "not robust to outliers"]
LIMITS = [
    "The sample size was small, so the estimates have wide intervals.",
    "All measurements came from a single laboratory and may not transfer to other settings.",
    "Several confounders could not be controlled, including {factor2}.",
    "The indexed corpus covers only a subset of the relevant literature, so absence of prior work is not evidence of novelty.",
    "We did not test values of {factor} outside the studied range.",
    "Measurement error in {prop} was not fully characterised.",
    "Model assumptions were not independently verified.",
    "Some documents could only be read through automatic text extraction, which may contain errors.",
]
VERIFY = [
    ("supported", "Verification: supported. The cited passage states the same quantity and direction of effect."),
    ("unsupported", "Verification: unsupported. No passage in the indexed corpus matches this claim."),
    ("partial", "Verification: partially supported. The passage reports a similar trend but a different magnitude."),
    ("contradicted", "Verification: contradicted. A cited passage reports the opposite direction of effect."),
]


def _num(rng: random.Random, lo: float, hi: float, digits: int = 1) -> str:
    return f"{rng.uniform(lo, hi):.{digits}f}"


def _generate_one(rng: random.Random, doc_no: int) -> str:
    domain = rng.choice(list(DOMAINS))
    d = DOMAINS[domain]
    subj, prop, prop2 = rng.choice(d["subjects"]), rng.choice(d["properties"]), rng.choice(d["properties"])
    method, method2 = rng.choice(d["methods"]), rng.choice(d["methods"])
    unit = rng.choice(d["units"])
    factor, factor2 = rng.sample(FACTORS, 2)
    design = rng.choice(DESIGNS)
    n1, n2 = rng.randint(12, 480), rng.randint(12, 480)
    year = rng.randint(1998, 2025)
    pct = _num(rng, 1.5, 48.0)
    direction = rng.choice(["increased", "decreased"])
    fmt = dict(subj=subj, prop=prop, prop2=prop2, method=method, method2=method2, unit=unit, factor=factor, factor2=factor2, n1=n1, n2=n2, pct=pct, domain=domain)

    out: list[str] = []
    out.append(f"Title: {prop[0].upper() + prop[1:]} of {subj}: a {design} study")
    out.append(f"Document: SYN-{doc_no:06d}    Field: {domain}    Year: {year}")
    out.append("")

    abstract = [
        f"We investigate how {prop} in {subj} varies with {factor}.",
        f"Using {method}, we measured {prop} across {n1} samples and compared them with {n2} control samples.",
        f"{prop[0].upper() + prop[1:]} {direction} by {pct} percent under the treatment condition, an effect that was {rng.choice(HEDGES)}.",
        f"We also examined {prop2} using {method2} to test whether the change was specific to {prop}.",
        f"These results suggest a link between {factor} and {prop}, but replication is required before any causal claim is made.",
    ]
    out.append("Abstract")
    out.append(" ".join(abstract[:2] + rng.sample(abstract[2:], rng.randint(1, 3))))
    out.append("")

    intro_pool = [
        f"Understanding {prop} in {subj} matters for {domain} because small changes can have large downstream effects.",
        f"Earlier studies reported that {prop} depends on {factor}, but the evidence was inconsistent across laboratories.",
        f"A {design} design allows {factor} to be varied while holding {factor2} fixed.",
        f"Prior work relied on {method}; here we combine it with {method2} to cross-check the result.",
        f"The goal of this study is to quantify the relationship between {factor} and {prop} in {subj}.",
        f"Whether {prop2} changes together with {prop} remains an open question.",
        f"We state two hypotheses: H1 predicts that {prop} {direction} with {factor}; H0 predicts no change.",
    ]
    out.append("1. Introduction")
    out.append(" ".join(rng.sample(intro_pool, rng.randint(4, 6))))
    out.append("")

    methods_pool = [
        f"We collected {n1} samples of {subj} and {n2} matched controls between {year - rng.randint(1, 3)} and {year}.",
        f"{method[0].upper() + method[1:]} was used to measure {prop}; each measurement was repeated {rng.randint(2, 6)} times.",
        f"{factor[0].upper() + factor[1:]} was varied in {rng.randint(4, 9)} steps from {_num(rng, 0.1, 5.0)} to {_num(rng, 6.0, 60.0)}.",
        f"Outliers beyond {_num(rng, 2.5, 4.0)} standard deviations were flagged and reported rather than removed.",
        f"Random seeds, software versions, and configuration files were stored with every run so that the analysis can be reproduced.",
        f"Statistical comparisons used a two-sided test with a significance level of {rng.choice(['0.05', '0.01', '0.001'])}.",
        f"{method2[0].upper() + method2[1:]} provided an independent check on {prop2}.",
        f"Missing values affected {_num(rng, 0.2, 6.0)} percent of records and were handled by listwise deletion.",
    ]
    out.append("2. Methods")
    out.append(" ".join(rng.sample(methods_pool, rng.randint(5, 8))))
    out.append("")

    mean_t, mean_c = rng.uniform(5, 90), rng.uniform(5, 90)
    results_pool = [
        f"The mean {prop} was {mean_t:.1f} {unit} in the treated group and {mean_c:.1f} {unit} in the control group.",
        f"The difference of {abs(mean_t - mean_c):.1f} {unit} had a {rng.choice([90, 95, 99])} percent confidence interval of {abs(mean_t - mean_c) - rng.uniform(0.5, 3):.1f} to {abs(mean_t - mean_c) + rng.uniform(0.5, 3):.1f}.",
        f"{prop[0].upper() + prop[1:]} {direction} monotonically with {factor} over the first {rng.randint(3, 7)} steps and then plateaued.",
        f"The correlation between {prop} and {prop2} was r = {rng.uniform(-0.9, 0.9):.2f} (n = {n1}).",
        f"Repeating the analysis with {method2} gave a similar estimate, within {_num(rng, 1, 15)} percent of the primary result.",
        f"Removing the {rng.randint(1, 5)} most extreme samples changed the estimated effect by {_num(rng, 0.1, 12.0)} percent.",
        f"No effect of {factor2} on {prop} was detected (p = {rng.uniform(0.06, 0.9):.2f}).",
        f"Variance between replicates was {_num(rng, 1, 25)} percent, which limits the precision of small effects.",
    ]
    out.append("3. Results")
    out.append(" ".join(rng.sample(results_pool, rng.randint(5, 8))))
    out.append("")

    out.append("4. Evidence ledger")
    for k in range(1, rng.randint(2, 4) + 1):
        status, sentence = rng.choice(VERIFY)
        out.append(f"Claim C{k}: {prop} of {subj} {direction} with {factor} (effect size {_num(rng, 0.1, 1.5, 2)}).")
        out.append(f"Evidence: document SYN-{rng.randint(1, 99999):06d}, page {rng.randint(1, 30)}, passage P{rng.randint(1, 60)}.")
        out.append(sentence)
        out.append(f"Status: {status}.")
    out.append("")

    out.append("5. Limitations")
    lim = rng.sample(LIMITS, rng.randint(3, 4))
    out.append(" ".join(s.format(**fmt) for s in lim))
    out.append("")

    out.append("6. Open questions")
    for q in rng.sample(
        [
            f"- Does the effect of {factor} on {prop} persist beyond the range tested here?",
            f"- Is {prop2} a mediator of the observed change in {prop}?",
            f"- Would {method2} reveal a different pattern in {subj} from another population?",
            f"- What sample size is needed to detect an effect of {_num(rng, 0.5, 5.0)} {unit}?",
            f"- Can an independent group reproduce the result using the stored configuration?",
        ],
        rng.randint(2, 4),
    ):
        out.append(q)
    out.append("")
    out.append(
        f"Conclusion. In this {design} study, {prop} in {subj} {direction} with {factor}. "
        f"The finding is {rng.choice(HEDGES)} and should be treated as a hypothesis until independently replicated."
    )
    return "\n".join(out) + "\n"


def generate_documents(n_docs: int, seed: int = 20260101) -> Iterator[tuple[str, str]]:
    """Yield (doc_id, text). Output is a pure function of (n_docs, seed)."""
    for i in range(n_docs):
        rng = random.Random(f"{seed}:{i}")
        yield f"syn-{i:06d}", _generate_one(rng, i)
